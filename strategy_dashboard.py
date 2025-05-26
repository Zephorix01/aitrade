
# --- Strategy Dashboard (Final Version) ---

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from alpaca_trade_api.rest import REST, TimeFrame, TimeFrameUnit
import smtplib, ssl
from email.message import EmailMessage
from streamlit_autorefresh import st_autorefresh

# --- Auto-Refresh ---
st_autorefresh(interval=60000, key="auto_refresh")

# --- Config (Secrets) ---
API_KEY = st.secrets["alpaca_api_key"]
SECRET_KEY = st.secrets["alpaca_secret_key"]
BASE_URL = st.secrets["alpaca_base_url"]

EMAIL_FROM = st.secrets["email_from"]
EMAIL_TO = st.secrets["email_to"]
EMAIL_PASS = st.secrets["email_pass"]
EMAIL_HOST = st.secrets.get("email_host", "smtp.gmail.com")
EMAIL_PORT = st.secrets.get("email_port", 465)

api = REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

# --- Utility Functions ---
def fetch_data(symbol, days=90):
    end = datetime.today()
    start = end - timedelta(days=days)
    bars = api.get_bars(
        symbol, TimeFrame(1, TimeFrameUnit.Day),
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        limit=days
    ).df
    bars = bars[bars['symbol'] == symbol]
    return bars

def get_latest_price(symbol):
    return api.get_latest_trade(symbol).price

def place_order(symbol, qty=1, side="buy"):
    try:
        api.submit_order(symbol=symbol, qty=qty, side=side, type='market', time_in_force='gtc')
        return f"{side.upper()} order placed for {symbol}"
    except Exception as e:
        return f"Order failed: {e}"

def send_email_alert(subject, body):
    try:
        msg = EmailMessage()
        msg.set_content(body)
        msg["Subject"] = subject
        msg["From"] = EMAIL_FROM
        msg["To"] = EMAIL_TO
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(EMAIL_HOST, EMAIL_PORT, context=context) as server:
            server.login(EMAIL_FROM, EMAIL_PASS)
            server.send_message(msg)
    except Exception as e:
        st.error(f"Email failed: {e}")

# --- Strategy Logic ---
def sma_crossover_strategy(df):
    df["sma10"] = df["close"].rolling(10).mean()
    df["sma30"] = df["close"].rolling(30).mean()
    signals = []
    for i in range(len(df)):
        if i < 30:
            signals.append("hold")
        elif df["sma10"].iloc[i] > df["sma30"].iloc[i] and df["sma10"].iloc[i-1] <= df["sma30"].iloc[i-1]:
            signals.append("buy")
        elif df["sma10"].iloc[i] < df["sma30"].iloc[i] and df["sma10"].iloc[i-1] >= df["sma30"].iloc[i-1]:
            signals.append("sell")
        else:
            signals.append("hold")
    return signals

def buy_and_hold_strategy(df):
    return ["buy"] + ["hold"] * (len(df)-1)

def rsi_strategy(df, period=14):
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    df["rsi"] = rsi
    signals = []
    for val in df["rsi"]:
        if pd.isna(val):
            signals.append("hold")
        elif val < 30:
            signals.append("buy")
        elif val > 70:
            signals.append("sell")
        else:
            signals.append("hold")
    return signals

def mean_reversion_strategy(df, window=20):
    df["mean"] = df["close"].rolling(window).mean()
    df["std"] = df["close"].rolling(window).std()
    signals = []
    for i in range(len(df)):
        if i < window or pd.isna(df["mean"].iloc[i]):
            signals.append("hold")
        elif df["close"].iloc[i] < df["mean"].iloc[i] - df["std"].iloc[i]:
            signals.append("buy")
        elif df["close"].iloc[i] > df["mean"].iloc[i] + df["std"].iloc[i]:
            signals.append("sell")
        else:
            signals.append("hold")
    return signals

# --- Streamlit UI ---
st.title("📈 AI Strategy Dashboard with Auto-Trading")
symbol = st.sidebar.text_input("Symbol", "AAPL")
days = st.sidebar.slider("Days", 30, 180, 90)

df = fetch_data(symbol, days)
st.line_chart(df["close"])

strategies = {
    "SMA Crossover": sma_crossover_strategy,
    "Buy and Hold": buy_and_hold_strategy,
    "RSI": rsi_strategy,
    "Mean Reversion": mean_reversion_strategy
}

selected = st.sidebar.radio("Strategy", list(strategies.keys()))
signals = strategies[selected](df)
df["Signal"] = signals
st.write(df[["close", "Signal"]].tail(20))

latest_signal = signals[-1]
st.subheader(f"📢 Latest Signal: `{latest_signal.upper()}`")

if st.sidebar.button("Place Trade Automatically"):
    if latest_signal in ["buy", "sell"]:
        result = place_order(symbol, qty=1, side=latest_signal)
        st.success(result)
        send_email_alert(
            subject=f"Trade Executed: {latest_signal.upper()} {symbol}",
            body=f"Executed {latest_signal} order for {symbol} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."
        )
    else:
        st.info("No trade signal at this time.")

