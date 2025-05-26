
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from alpaca_trade_api.rest import REST, TimeFrame, TimeFrameUnit
import smtplib, ssl
from email.message import EmailMessage

# --- Auto Refresh ---
st_autorefresh(interval=60000, key="auto_refresh")

# --- API Setup ---
API_KEY = st.secrets["alpaca_api_key"]
SECRET_KEY = st.secrets["alpaca_secret_key"]
BASE_URL = "https://paper-api.alpaca.markets"
api = REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

# --- Utility Functions ---
def fetch_data(symbol, days=90):
    end = datetime.today()
    start = end - timedelta(days=days)
    try:
        bars = api.get_bars(symbol, TimeFrame(1, TimeFrameUnit.Day), start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), limit=days, feed='iex')
        df = bars.df
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def get_latest_price(symbol):
    try:
        trade = api.get_latest_trade(symbol)
        return trade.price
    except Exception as e:
        st.error(f"Failed to retrieve price: {e}")
        return None

def send_email_alert(subject, body):
    try:
        msg = EmailMessage()
        msg.set_content(body)
        msg["Subject"] = subject
        msg["From"] = st.secrets["email_from"]
        msg["To"] = st.secrets["email_to"]
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(st.secrets["email_host"], st.secrets["email_port"], context=context) as server:
            server.login(st.secrets["email_from"], st.secrets["email_pass"])
            server.send_message(msg)
    except Exception as e:
        st.sidebar.error(f"Email failed: {e}")

# --- Strategies ---
def sma_crossover_strategy(df):
    df["sma_short"] = df["close"].rolling(window=10).mean()
    df["sma_long"] = df["close"].rolling(window=30).mean()
    signals = []
    for i in range(len(df)):
        if i == 0 or pd.isna(df["sma_short"].iloc[i]) or pd.isna(df["sma_long"].iloc[i]):
            signals.append("hold")
        elif df["sma_short"].iloc[i] > df["sma_long"].iloc[i] and df["sma_short"].iloc[i-1] <= df["sma_long"].iloc[i-1]:
            signals.append("buy")
        elif df["sma_short"].iloc[i] < df["sma_long"].iloc[i] and df["sma_short"].iloc[i-1] >= df["sma_long"].iloc[i-1]:
            signals.append("sell")
        else:
            signals.append("hold")
    return signals

def buy_and_hold_strategy(df):
    return ["buy"] + ["hold"] * (len(df) - 1)

def rsi_strategy(df, period=14):
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    signals = []
    for val in rsi:
        if pd.isna(val):
            signals.append("hold")
        elif val < 30:
            signals.append("buy")
        elif val > 70:
            signals.append("sell")
        else:
            signals.append("hold")
    return signals

def momentum_strategy(df, period=10):
    momentum = df["close"].diff(period)
    signals = []
    for m in momentum:
        if pd.isna(m):
            signals.append("hold")
        elif m > 0:
            signals.append("buy")
        elif m < 0:
            signals.append("sell")
        else:
            signals.append("hold")
    return signals

def mean_reversion_strategy(df, window=20):
    df["mean"] = df["close"].rolling(window=window).mean()
    df["std"] = df["close"].rolling(window=window).std()
    signals = []
    for i in range(len(df)):
        if i == 0 or pd.isna(df["mean"].iloc[i]) or pd.isna(df["std"].iloc[i]):
            signals.append("hold")
        elif df["close"].iloc[i] < df["mean"].iloc[i] - df["std"].iloc[i]:
            signals.append("buy")
        elif df["close"].iloc[i] > df["mean"].iloc[i] + df["std"].iloc[i]:
            signals.append("sell")
        else:
            signals.append("hold")
    return signals

# --- Streamlit UI ---
st.title("📈 AI Strategy Backtester")
symbol = st.sidebar.text_input("Symbol", value="AAPL")
days = st.sidebar.slider("Days of Data", 30, 180, 90)

strategies = {
    "SMA Crossover": sma_crossover_strategy,
    "Buy & Hold": buy_and_hold_strategy,
    "RSI": rsi_strategy,
    "Momentum": momentum_strategy,
    "Mean Reversion": mean_reversion_strategy
}
selected = st.sidebar.multiselect("Select Strategies", strategies.keys(), default=list(strategies.keys()))

if st.sidebar.button("Run Backtest"):
    df = fetch_data(symbol, days)
    if df.empty:
        st.warning("No data fetched.")
    else:
        for name in selected:
            st.subheader(name)
            sig_func = strategies[name]
            signals = sig_func(df)
            value = [10000]
            for signal in signals:
                if signal == "buy":
                    value.append(value[-1] * 1.01)
                elif signal == "sell":
                    value.append(value[-1] * 0.99)
                else:
                    value.append(value[-1])
            fig, ax = plt.subplots()
            ax.plot(value)
            ax.set_title(f"{name} Strategy Portfolio Value")
            st.pyplot(fig)

# --- Alerts ---
st.sidebar.subheader("📣 Price Alert")
alert_symbol = st.sidebar.text_input("Watch Symbol", value="AAPL", key="alert_symbol")
alert_price = st.sidebar.number_input("Target Price", value=100.0, step=0.1, key="alert_price")

if st.sidebar.button("Activate Alert"):
    st.session_state["alert"] = {"symbol": alert_symbol, "price": alert_price}

if "alert" in st.session_state:
    info = st.session_state["alert"]
    latest = get_latest_price(info["symbol"])
    if latest:
        st.sidebar.write(f"{info['symbol']} is at ${latest:.2f} (Target: ${info['price']})")
        if latest >= info["price"]:
            st.sidebar.success("✅ Target hit!")
            send_email_alert(f"{info['symbol']} Alert", f"Price hit ${latest:.2f} (target ${info['price']})")
