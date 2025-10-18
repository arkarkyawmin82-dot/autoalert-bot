# -*- coding: utf-8 -*-
"""
tg_alertbot_v2_safe_tp_fix.py
Same as TP version, but FIXES the ccxt wrapper closure bug that caused:
"binance.fetch_trades() got an unexpected keyword argument 'timeframe'".
"""

import os, time, requests, math, threading
from datetime import datetime
import numpy as np
import pandas as pd
import ccxt

# ================= USER CONFIG =================
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "").strip()
TG_CHAT_ID   = os.getenv("TG_CHAT_ID", "").strip()
POLL_SEC     = 30
STRICT_MODE  = True
MIN_SCORE    = 3
VOL_MULT     = 1.4
MIN_ATR_PCT  = 0.02 / 100.0
MAX_EXT_ATR  = 2.0

UT = {"15m": {"ATR": 1.0, "KEY": 2.0},
      "1h":  {"ATR": 2.0, "KEY": 3.0},
      "4h":  {"ATR": 2.0, "KEY": 3.0}}

ATR_TP_MULTS = (1.0, 2.0, 3.0)
SL_LONG_MULT  = 2.0
SL_SHORT_MULT = 2.0
SL_EMA_BUFF   = 0.5

WATCH = [
    "SOLUSDT", "XRPUSDT", "BNBUSDT", "ETHUSDT", "BTCUSDT",
    "ASTRUSDT", "AVAXUSDT", "LINKUSDT"
]

# ================= EXCHANGE CLIENTS (PUBLIC) =================
futures = ccxt.binanceusdm({"enableRateLimit": True})
spot    = ccxt.binance({"enableRateLimit": True})

def _norm_usdm_symbol(s):
    s = str(s).strip().upper().replace("PERP","")
    if "/" not in s:
        s = f"{s[:-4]}/USDT" if s.endswith("USDT") else f"{s}/USDT"
    if ":USDT" not in s:
        s = s + ":USDT"
    return s

def _wrap_ccxt_symbol_methods():
    # market() wrapper (simple)
    try:
        _orig_market = futures.market
        def _market_wrap(symbol, _orig=_orig_market):
            return _orig(_norm_usdm_symbol(symbol))
        futures.market = _market_wrap
    except Exception:
        pass

    # Safe factory that binds "orig" per method
    def _make_wrapper(orig_func):
        def _wrapped(symbol, *args, **kwargs):
            return orig_func(_norm_usdm_symbol(symbol), *args, **kwargs)
        return _wrapped

    for name in ["fetch_ohlcv", "fetch_ticker", "fetch_order_book", "fetch_trades"]:
        try:
            orig = getattr(futures, name)
            setattr(futures, name, _make_wrapper(orig))
        except Exception:
            pass

_wrap_ccxt_symbol_methods()

# ================= INDICATORS =================
def _ema(a: pd.Series, n: int):
    return a.ewm(span=n, adjust=False).mean()

def _rma(series: pd.Series, n: int):
    return series.ewm(alpha=1.0/n, adjust=False).mean()

def _atr(df: pd.DataFrame, n: int = 14):
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return _rma(tr, n)

def _rsi(close: pd.Series, n: int = 14):
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/n, adjust=False).mean()
    roll_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def _macd(close: pd.Series, fast=12, slow=26, signal=9):
    fast_ema = _ema(close, fast)
    slow_ema = _ema(close, slow)
    line = fast_ema - slow_ema
    sig  = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return line, sig, hist

def _stoch_kd(df: pd.DataFrame, k_len=14, d_len=3):
    hh = df["high"].rolling(k_len).max()
    ll = df["low"].rolling(k_len).min()
    k = (df["close"] - ll) * 100.0 / (hh - ll + 1e-12)
    d = k.rolling(d_len).mean()
    return k, d

def _vol_ma(vol: pd.Series, n=20):
    return vol.rolling(n).mean()

def _trendline_break(df: pd.DataFrame, lookback=60):
    if len(df) < lookback + 5:
        return False, False
    seg = df.tail(lookback).copy()
    x = np.arange(len(seg))
    up_coef = np.polyfit(x, seg["high"].values, 1)
    lo_coef = np.polyfit(x, seg["low"].values, 1)
    up_line = up_coef[0]*x + up_coef[1]
    lo_line = lo_coef[0]*x + lo_coef[1]
    last_close = float(seg["close"].iloc[-1])
    return last_close > up_line[-1], last_close < lo_line[-1]

def _ohlcv_usdm(sym: str, tf: str, lim=300) -> pd.DataFrame:
    data = futures.fetch_ohlcv(sym, timeframe=tf, limit=lim)
    df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
    return df

def _prepare(df: pd.DataFrame):
    c = df["close"]
    df["ema9"]   = _ema(c, 9)
    df["ema21"]  = _ema(c, 21)
    df["ema50"]  = _ema(c, 50)
    df["ema200"] = _ema(c, 200)
    df["atr"]    = _atr(df, 14)
    df["rsi"]    = _rsi(c, 14)
    macd_line, macd_sig, macd_hist = _macd(c, 12, 26, 9)
    df["macd_l"], df["macd_s"], df["macd_h"] = macd_line, macd_sig, macd_hist
    k, d = _stoch_kd(df, 14, 3)
    df["stoch_k"], df["stoch_d"] = k, d
    df["vol_ma20"] = _vol_ma(df["volume"], 20)
    return df

def _ut_bot_signal(df: pd.DataFrame, key_value=2.0, atr_mult=1.0):
    c = df["close"]
    atr = _atr(df, 14)
    baseline = _ema(c, int(10 * key_value))
    up_trail   = baseline - atr * atr_mult
    down_trail = baseline + atr * atr_mult
    buy  = (c.shift(1) <= down_trail.shift(1)) & (c > down_trail)
    sell = (c.shift(1) >= up_trail.shift(1))   & (c < up_trail)
    return bool(buy.iloc[-1]), bool(sell.iloc[-1]), float(atr.iloc[-1])

# ================= SCORING & FILTERS =================
def _score_side(closed):
    price = float(closed["close"])
    ema9, ema21, ema50, ema200 = map(float, [closed["ema9"], closed["ema21"], closed["ema50"], closed["ema200"]])
    rsi  = float(closed["rsi"])
    mh, ml, ms = float(closed["macd_h"]), float(closed["macd_l"]), float(closed["macd_s"])
    k, d = float(closed["stoch_k"]), float(closed["stoch_d"])
    v, vma = float(closed["volume"]), float(closed["vol_ma20"]) if not math.isnan(closed["vol_ma20"]) else 0.0

    long_score = short_score = 0
    if ema9 > ema21 > ema50 > ema200: long_score += 2
    if ema9 < ema21 < ema50 < ema200: short_score += 2
    if price > ema21: long_score += 1
    if price < ema21: short_score += 1
    if 52 <= rsi <= 68: long_score += 1
    if 32 <= rsi <= 48: short_score += 1
    if mh > 0 and ml > ms: long_score += 1
    if mh < 0 and ml < ms: short_score += 1
    if 20 < k < 80 and k > d: long_score += 1
    if 20 < k < 80 and k < d: short_score += 1
    if vma > 0:
        vspike = v > VOL_MULT * vma
        if vspike and price > ema21: long_score += 1
        if vspike and price < ema21: short_score += 1
    return long_score, short_score

def _extra_guards(closed):
    price = float(closed["close"])
    ema21 = float(closed["ema21"])
    atr   = float(closed["atr"]) if not math.isnan(closed["atr"]) else 0.0
    rsi   = float(closed["rsi"])
    mh    = float(closed["macd_h"])
    if price > 0 and atr / price < MIN_ATR_PCT:
        return False, False, "ATR too small"
    if abs(price - ema21) > MAX_EXT_ATR * atr:
        return False, False, "Overextended vs EMA21"
    ok_long  = 40 <= rsi <= 68 and mh >= 0
    ok_short = 32 <= rsi <= 60 and mh <= 0
    why = None
    if not ok_long and not ok_short:
        why = "RSI/MACD guardrail"
    return ok_long, ok_short, why

def _edge_and_side(df: pd.DataFrame):
    if len(df) < 120:
        return None, None, "not_enough_data"
    closed = df.iloc[-2]
    l, s = _score_side(closed)
    edge = l - s
    side = "LONG" if edge >= MIN_SCORE and l > s else ("SHORT" if -edge >= MIN_SCORE and s > l else "FLAT")
    ok_long, ok_short, _ = _extra_guards(closed)
    if side == "LONG" and not ok_long: side = "FLAT"
    if side == "SHORT" and not ok_short: side = "FLAT"
    return side, closed, None

# ================= SL/TP HELPERS =================
def _fmt(x):
    try: return f"{float(x):.2f}"
    except: return str(x)

def _levels(side, entry, atr, ema21):
    if side == "LONG":
        sl1 = entry - SL_LONG_MULT*atr
        sl2 = ema21 - SL_EMA_BUFF*atr
        sl  = min(sl1, sl2)
        tps = [entry + k*atr for k in ATR_TP_MULTS]
    else:
        sl1 = entry + SL_SHORT_MULT*atr
        sl2 = ema21 + SL_EMA_BUFF*atr
        sl  = max(sl1, sl2)
        tps = [entry - k*atr for k in ATR_TP_MULTS]
    return sl, tps[0], tps[1], tps[2]

# ================= MULTI-TF CONFIRMATION =================
def _confirmed_3tf(symbol, key_value=2.0, atr_mult=1.0, strict=True):
    s = symbol.upper()
    if "/" not in s:
        s = s if s.endswith("USDT") else s + "USDT"
        s = s[:-4] + "/USDT:USDT"
    try:
        m15 = _prepare(_ohlcv_usdm(s, "15m", 400))
        h1  = _prepare(_ohlcv_usdm(s, "1h",  400))
        h4  = _prepare(_ohlcv_usdm(s, "4h",  400))
    except Exception as e:
        return None, None, f"data_error: {e}"
    side15, c15, _ = _edge_and_side(m15)
    side1,  _,   _ = _edge_and_side(h1)
    side4,  _,   _ = _edge_and_side(h4)
    ut_buy_15, ut_sell_15, atr_last_15 = _ut_bot_signal(m15, key_value, atr_mult)
    decision = "⚠️ No entry — needs alignment (15m must match 1h & 4h)"
    final = "FLAT"
    if side15 in ("LONG", "SHORT"):
        if strict:
            if side1 == side15 and side4 == side15:
                final = side15
                decision = f"✅✅✅ CONFIRMED {final} — 15m aligned with 1h & 4h"
        else:
            higher = [side1, side4]
            if (side15 in higher) or (all(x in ("FLAT", side15) for x in higher)):
                final = side15
                decision = f"✅ CONFIRMED (soft) {final} — No opposite on 1h/4h"
    if c15 is None:
        return None, None, "no_data"
    price = float(c15["close"])
    atr   = float(c15["atr"])
    ema9, ema21, ema50, ema200 = map(float, [c15["ema9"], c15["ema21"], c15["ema50"], c15["ema200"]])
    rsi  = float(c15["rsi"])
    mh, ml, ms = map(float, [c15["macd_h"], c15["macd_l"], c15["macd_s"]])
    k, d = map(float, [c15["stoch_k"], c15["stoch_d"]])
    lines = [
        f"🧭 15m Entry + 1h/4h Confirm — {symbol}",
        f"Decision: {decision}", "",
        "── 15m ──",
        f"Side: {side15}",
        f"Price: {_fmt(price)} | ATR(14): {_fmt(atr)}",
        f"EMA9/21/50/200: {_fmt(ema9)}, {_fmt(ema21)}, {_fmt(ema50)}, {_fmt(ema200)}",
        f"RSI(14): {_fmt(rsi)} | MACD(h/l/s): {_fmt(mh)}/{_fmt(ml)}/{_fmt(ms)}",
        f"Stoch K/D: {_fmt(k)}/{_fmt(d)} | UT: buy={ut_buy_15} sell={ut_sell_15}", "",
        "── 1h ──", f"Side: {side1}",
        "── 4h ──", f"Side: {side4}",
    ]
    if final in ("LONG", "SHORT"):
        sl, tp1, tp2, tp3 = _levels(final, price, atr, ema21)
        lines += ["", f"Entry: {_fmt(price)}", f"SL:    {_fmt(sl)}",
                  f"TP1:   {_fmt(tp1)}", f"TP2:   {_fmt(tp2)}", f"TP3:   {_fmt(tp3)}"]
    return final, "\n".join(lines), None

# ================= TELEGRAM HELPERS =================
def tg_send(text: str):
    if not (TG_BOT_TOKEN and TG_CHAT_ID):
        print("[TG_DISABLED]", text[:2000]); return
    try:
        requests.get(f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage",
                     params={"chat_id": TG_CHAT_ID, "text": text[:3900]}, timeout=10)
    except Exception as e:
        print("tg error:", e)

# ================= COMMANDS & AUTO-SCAN =================
RUNNING = True
AUTO_SCAN = True
last_signal = {}

HELP_TEXT = "\n".join([
    "Commands:",
    "/help – Show this help",
    "/watchlist – Show default coins",
    "/status – Bot status",
    "/pause – Pause auto scan",
    "/resume – Resume auto scan",
    "/analyze SYMBOL – 3TF confirm (e.g., /analyze SOLUSDT)",
    "/strict on|off – Toggle strict mode (default: on)",
])

def handle_command(text: str):
    global RUNNING, AUTO_SCAN, STRICT_MODE
    low = text.strip().lower()
    if low.startswith("/help"):
        tg_send(HELP_TEXT)
    elif low.startswith("/watchlist"):
        tg_send("Watchlist: " + ", ".join(WATCH))
    elif low.startswith("/status"):
        tg_send(f"running={RUNNING} auto_scan={AUTO_SCAN} strict={STRICT_MODE}\nlast={last_signal}")
    elif low.startswith("/pause"):
        AUTO_SCAN = False; tg_send("⏸️ Auto-scan paused.")
    elif low.startswith("/resume"):
        AUTO_SCAN = True; tg_send("▶️ Auto-scan resumed.")
    elif low.startswith("/strict"):
        parts = text.split()
        if len(parts) >= 2: STRICT_MODE = parts[1].lower() == "on"
        tg_send(f"STRICT mode = {STRICT_MODE}")
    elif low.startswith("/analyze"):
        parts = text.split()
        if len(parts) < 2: tg_send("Usage: /analyze SYMBOL"); return
        sym = parts[1].upper()
        try:
            side, msg, err = _confirmed_3tf(sym, key_value=UT['15m']['KEY'], atr_mult=UT['15m']['ATR'], strict=STRICT_MODE)
            if err: tg_send(f"analyze error: {err}")
            else:   tg_send(msg)
        except Exception as e:
            tg_send(f"analyze error: {e}")

def tg_poll_commands():
    if not (TG_BOT_TOKEN and TG_CHAT_ID):
        print("Telegram disabled (no token/chat_id)."); return
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/getUpdates"
    send_url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    offset = None
    try: requests.get(send_url, params={"chat_id": TG_CHAT_ID, "text": "🤖 AlertBot online. Use /help"}, timeout=10)
    except Exception: pass
    while True:
        try:
            params = {"timeout": 30}
            if offset: params["offset"] = offset
            r = requests.get(url, params=params, timeout=35).json()
            if not r.get("ok"): time.sleep(2); continue
            for up in r.get("result", []):
                offset = up["update_id"] + 1
                msg = up.get("message") or up.get("edited_message")
                if not msg: continue
                chat_id = str(msg["chat"]["id"])
                if chat_id != str(TG_CHAT_ID): continue
                text = (msg.get("text") or "").strip()
                if not text: continue
                handle_command(text)
        except Exception as e:
            print("Command poll error:", e); time.sleep(3)

def auto_scan_loop():
    tg_send("🛰️ Auto-scan running…")
    while True:
        if not AUTO_SCAN: time.sleep(1); continue
        for sym in WATCH:
            try:
                side, msg, err = _confirmed_3tf(sym, key_value=UT['15m']['KEY'], atr_mult=UT['15m']['ATR'], strict=STRICT_MODE)
                if err: print(f"[{sym}] skip: {err}"); continue
                prev = last_signal.get(sym, "FLAT")
                if side in ("LONG", "SHORT") and side != prev:
                    last_signal[sym] = side; tg_send(msg)
                elif side == "FLAT":
                    last_signal[sym] = "FLAT"
            except Exception as e:
                print(f"[{sym}] scan error:", e); continue
        time.sleep(POLL_SEC)

def main():
    print("🤖 Bot started. (FIX applied)")
    t1 = threading.Thread(target=tg_poll_commands, daemon=True); t1.start()
    t2 = threading.Thread(target=auto_scan_loop, daemon=True); t2.start()
    while True: time.sleep(60)

if __name__ == "__main__":
    main()
