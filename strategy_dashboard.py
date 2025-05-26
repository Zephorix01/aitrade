import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import smtplib, ssl
from email.message import EmailMessage
from alpaca_trade_api.rest import REST, TimeFrame, TimeFrameUnit

# --- Auto-Refresh ---
st_autorefresh(interval=60000, key="auto_refresh")

# --- Config (from secrets) ---
API_KEY = st.secrets["alpaca_api_key"]
SECRET_KEY = st.secrets["alpaca_secret_key"]
BASE_URL = st.secrets["alpaca_base_url"]
EMAIL_FROM = st.secrets["email_from"]
EMAIL_TO = st.secrets["email_to"]
EMAIL_PASS = st.secrets["email_pass"]
EMAIL_HOST = st.secrets.get("email_host", "smtp.gmail.com")
EMAIL_PORT = st.secrets.get("email_port", 465)

api = REST(API_KEY, SECRET_KEY, BASE_URL, api_version="v2")

# --- Helper Functions ---
def fetch_data(symbol, days=90):
    end = datetime.today()
    start = end - timedelta(days=days)
    bars = api.get_bars(symbol, TimeFrame(1, TimeFrameUnit.Day), start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), limit=days, feed='iex')
    df = bars.df
    df = df[df['symbol'] == symbol]
    return df

def get_latest_price(symbol):
    return api.get_latest_trade(symbol).price

def place_paper_trade(symbol, qty=1, side="buy"):
    try:
        api.submit_order(symbol=symbol, qty=qty, side=side, type="market", time_in_force="gtc")
        return f"✅ {side.upper()} order placed for {symbol}."
    except Exception as e:
        return f"❌ Trade error: {e}"

def send_email_alert(subject, body):
    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(EMAIL_HOST, EMAIL_PORT, context=context) as server:
        server.login(EMAIL_FROM, EMAIL_PASS)
        server.send_message(msg)

def track_rl_weights(weights):
    df = pd.DataFrame.from_dict(weights, orient='index', columns=['Weight'])
    df = df.sort_values(by='Weight', ascending=False)
    st.subheader("🧠 RL Strategy Weights")
    st.bar_chart(df)

# --- Strategies ---
def sma_crossover_strategy(df):
    df["sma_short"] = df["close"].rolling(10).mean()
    df["sma_long"] = df["close"].rolling(30).mean()
    signals = []
    for i in range(len(df)):
        if i == 0 or pd.isna(df["sma_short"].iloc[i]) or pd.isna(df["sma_long"].iloc[i]):
            signals.append("hold")
        elif df["sma_short"].iloc[i] > df["sma_long"].iloc[i] and df["sma_short"].iloc[i - 1] <= df["sma_long"].iloc[i - 1]:
            signals.append("buy")
        elif df["sma_short"].iloc[i] < df["sma_long"].iloc[i] and df["sma_short"].iloc[i - 1] >= df["sma_long"].iloc[i - 1]:
            signals.append("sell")
        else:
            signals.append("hold")
    return signals

def buy_and_hold_strategy(df):
    return ["buy"] + ["hold"] * (len(df) - 1)

def rsi_strategy(df, period=14):
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    df["rsi"] = rsi
    signals = []
    for val in rsi:
        if pd.isna(val): signals.append("hold")
        elif val < 30: signals.append("buy")
        elif val > 70: signals.append("sell")
        else: signals.append("hold")
    return signals

def momentum_strategy(df, period=10):
    momentum = df["close"].diff(period)
    signals = []
    for m in momentum:
        if pd.isna(m): signals.append("hold")
        elif m > 0: signals.append("buy")
        elif m < 0: signals.append("sell")
        else: signals.append("hold")
    return signals

def mean_reversion_strategy(df, window=20):
    mean = df["close"].rolling(window).mean()
    std = df["close"].rolling(window).std()
    signals = []
    for i in range(len(df)):
        if pd.isna(mean[i]) or pd.isna(std[i]):
            signals.append("hold")
        elif df["close"][i] < mean[i] - std[i]:
            signals.append("buy")
        elif df["close"][i] > mean[i] + std[i]:
            signals.append("sell")
        else:
            signals.append("hold")
    return signals

# --- Backtest ---
def backtest_strategy(df, signals):
    portfolio = [10000]
    for sig in signals:
        if sig == "buy": portfolio.append(portfolio[-1] * 1.01)
        elif sig == "sell": portfolio.append(portfolio[-1] * 0.99)
        else: portfolio.append(portfolio[-1])
    return portfolio

# --- UI ---
st.title("📊 AI Strategy Dashboard")

symbol = st.sidebar.text_input("Symbol", value="AAPL")
days = st.sidebar.slider("Days", 30, 180, 90)

strategies = {
    "SMA Crossover": sma_crossover_strategy,
    "Buy & Hold": buy_and_hold_strategy,
    "RSI Strategy": rsi_strategy,
    "Momentum": momentum_strategy,
    "Mean Reversion": mean_reversion_strategy
}

selected_strategies = st.sidebar.multiselect("Choose Strategies", list(strategies.keys()), default=list(strategies.keys()))

df = fetch_data(symbol, days)
if df.empty:
    st.warning("No data found.")
    st.stop()

best_value = 0
best_strategy = None

for name in selected_strategies:
    st.subheader(f"📈 {name}")
    signals = strategies[name](df)
    result = backtest_strategy(df, signals)
    final_val = result[-1]
    if final_val > best_value:
        best_value = final_val
        best_strategy = name
    st.line_chart(result)

st.success(f"🏆 Best Strategy: {best_strategy}")
track_rl_weights({s: backtest_strategy(df, strategies[s](df))[-1] for s in selected_strategies})

# --- Alerts ---
st.sidebar.subheader("📣 Price Alert")
alert_symbol = st.sidebar.text_input("Alert Symbol", value=symbol, key="alert_symbol")
target_price = st.sidebar.number_input("Target Price", value=100.0, step=0.1, key="alert_price")
if st.sidebar.button("Check Alert"):
    current = get_latest_price(alert_symbol)
    st.sidebar.write(f"Current: ${current:.2f}")
    if current >= target_price:
        st.sidebar.success("🎯 Target reached!")
        send_email_alert(f"{alert_symbol} Hit Alert", f"{alert_symbol} hit ${current:.2f}")
    else:
        st.sidebar.info("Target not reached.")

# --- Auto-Trading ---
st.sidebar.subheader("🚀 Auto-Trading")
auto_enabled = st.sidebar.checkbox("Enable Auto-Trade")
if auto_enabled and best_strategy:
    sigs = strategies[best_strategy](df)
    latest_sig = sigs[-1]
    if latest_sig in ["buy", "sell"]:
        result = place_paper_trade(symbol, side=latest_sig)
        st.sidebar.success(result)
    else:
        st.sidebar.info("No action taken.")
