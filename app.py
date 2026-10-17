import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Signal Bot", page_icon="📈")
st.title("📈 Signal Bot")
st.caption("Signals only. Not financial advice. Always check the backtest first.")

symbol = st.text_input("Asset (Binance symbol)", "BTCUSDT").upper()
timeframe = st.selectbox("Candle size", ["1m", "3m", "5m", "15m"])
payout = st.slider("Payout %", 70, 95, 85) / 100
min_score = st.selectbox("Signal strictness (higher = fewer trades)", [2, 3], index=0)


def get_data(sym, tf):
    for base in ["https://data-api.binance.vision", "https://api.binance.com"]:
        try:
            raw = requests.get(base + "/api/v3/klines",
                               params={"symbol": sym, "interval": tf, "limit": 1000},
                               timeout=10).json()
            df = pd.DataFrame(raw, columns=["time", "open", "high", "low", "close", "vol",
                                            "ct", "qv", "n", "tb", "tq", "i"])
            df["close"] = df["close"].astype(float)
            return df[["time", "close"]]
        except Exception:
            continue
    return None


def add_indicators(df):
    c = df["close"]
    df["ema_fast"] = c.ewm(span=9, adjust=False).mean()
    df["ema_slow"] = c.ewm(span=21, adjust=False).mean()
    d = c.diff()
    g = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + g / l.replace(0, np.nan))
    m = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    df["macd_hist"] = m - m.ewm(span=9, adjust=False).mean()
    return df.dropna().reset_index(drop=True)


def score(r):
    s = 1 if r.ema_fast > r.ema_slow else -1
    s += 1 if r.macd_hist > 0 else -1
    if r.rsi < 30:
        s += 1
    elif r.rsi > 70:
        s -= 1
    return s


def to_signal(s):
    return "BUY" if s >= min_score else "SELL" if s <= -min_score else "WAIT"


if st.button("🔄 Get signal now", type="primary"):
    data = get_data(symbol, timeframe)
    if data is None:
        st.error("Could not get data. Check the asset name and try again.")
    else:
        df = add_indicators(data)
        df["sig"] = df.apply(lambda r: to_signal(score(r)), axis=1)
        last = df.iloc[-1]

        color = {"BUY": "green", "SELL": "red", "WAIT": "gray"}[last.sig]
        st.markdown(f"<h1 style='text-align:center;color:{color}'>{last.sig}</h1>",
                    unsafe_allow_html=True)
        st.write(f"Price: {last.close:.4f}  |  RSI: {last.rsi:.1f}")

        wins = losses = 0
        pnl = 0.0
        for i in range(len(df) - 1):
            if df.at[i, "sig"] == "WAIT":
                continue
            a, b = df.at[i, "close"], df.at[i + 1, "close"]
            if a == b:
                continue
            won = b > a if df.at[i, "sig"] == "BUY" else b < a
            wins += won
            losses += (not won)
            pnl += payout if won else -1

        st.subheader("Backtest (last 1000 candles)")
        n = wins + losses
        if n:
            wr = wins / n * 100
            be = 1 / (1 + payout) * 100
            st.write(f"Trades: {n}  |  Win rate: {wr:.1f}%  |  Break-even needed: {be:.1f}%")
            if pnl > 0:
                st.success(f"Profit: {pnl:+.2f} units")
            else:
                st.warning(f"Loss: {pnl:+.2f} units. This strategy is losing on this data.")
        else:
            st.write("No trades generated.")
