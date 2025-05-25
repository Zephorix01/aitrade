import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import csv
import random
import itertools
from datetime import datetime, timedelta
from alpaca_trade_api.rest import REST, TimeFrame
from alpaca_trade_api.rest import TimeFrameUnit

# --- Config ---
API_KEY = "PKUY243PPUK29FFEBO0W"
SECRET_KEY = "buEgMnp7k6GoreDYy7SC63zHV4YCErrKy9EKzCJz"
BASE_URL = "https://paper-api.alpaca.markets"
api = REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

# --- Utility Functions ---
def fetch_data(symbol, days=90):
    end = datetime.today()
    start = end - timedelta(days=days)
    try:
        bars = api.get_bars(symbol, TimeFrame(1, TimeFrameUnit.Day), start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), limit=days, feed='iex')
        return bars
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return []

def place_paper_trade(symbol, qty, side="buy"):
    try:
        api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='market',
            time_in_force='gtc'
        )
        return f"✅ {side.upper()} order for {qty} shares of {symbol} placed."
    except Exception as e:
        return f"❌ Failed to place order: {e}"

def track_rl_weights(weights):
    df = pd.DataFrame.from_dict(weights, orient='index', columns=['Weight'])
    df = df.sort_values(by='Weight', ascending=False)
    st.subheader("📊 Reinforcement Learning Strategy Weights")
    st.bar_chart(df)

# --- Daily Auto-Retrain and Logging ---
def auto_daily_retrain(symbol="AAPL", days=90):
    from pathlib import Path
    Path("logs").mkdir(exist_ok=True)
    bars = fetch_data(symbol, days)

    from strategy_dashboard import (
        sma_crossover_strategy, buy_and_hold_strategy, rsi_strategy,
        momentum_strategy, mean_reversion_strategy, macd_strategy,
        bollinger_strategy, ema_strategy, backtest_strategy,
        reinforcement_learn, load_model, save_model
    )

    strategies = {
        "SMA Crossover": sma_crossover_strategy,
        "Buy & Hold": buy_and_hold_strategy,
        "RSI": rsi_strategy,
        "Momentum": momentum_strategy,
        "Mean Reversion": mean_reversion_strategy,
        "MACD": macd_strategy,
        "Bollinger Bands": bollinger_strategy,
        "EMA Crossover": ema_strategy
    }

    all_metrics = {}
    for name, strategy in strategies.items():
        signals = strategy(bars)
        _, _, metrics = backtest_strategy(bars, signals)
        all_metrics[name] = metrics

    weights = load_model()
    updated = reinforcement_learn(weights, all_metrics)
    save_model({"weights": updated})

    log_file = f"logs/log_{datetime.today().strftime('%Y-%m-%d')}.csv"
    df = pd.DataFrame.from_dict(all_metrics, orient='index')
    df["date"] = datetime.today().strftime('%Y-%m-%d')
    df.to_csv(log_file, index=True)
    return log_file

# --- Optional Email Alerts ---
def send_email_alert(subject, body):
    import smtplib, ssl
    from email.message import EmailMessage

    EMAIL_FROM = "reece.flew@gmail.com"
    EMAIL_TO = "reece.flew@gmail.com"
    EMAIL_PASS = "Hq1569801"
    EMAIL_HOST = "smtp.gmail.com"
    EMAIL_PORT = 465

    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(EMAIL_HOST, EMAIL_PORT, context=context) as server:
            server.login(EMAIL_FROM, EMAIL_PASS)
            server.send_message(msg)
        st.sidebar.success("✅ Email summary sent.")
    except Exception as e:
        st.sidebar.error(f"❌ Email send failed: {e}")

# --- Run retrain and email if checkbox is active ---
if st.sidebar.checkbox("📅 Run Daily Retrain + Email Summary"):
    log_file = auto_daily_retrain()
    st.sidebar.success(f"Daily retrain complete. Log: {log_file}")
    if st.sidebar.checkbox("✉️ Send Email Summary"):
        with open(log_file, "r") as f:
            summary = f.read()
        send_email_alert("Daily Trading Strategy Summary", summary)

# Keep the rest of the existing code as-is below...

# --- Live Trading Controls ---
st.sidebar.subheader("📤 Live Trade Simulation")
trade_symbol = st.sidebar.text_input("Trade Symbol", "AAPL")
trade_qty = st.sidebar.number_input("Quantity", min_value=1, value=10)

col1, col2 = st.sidebar.columns(2)
if col1.button("Buy Now"):
    message = place_paper_trade(trade_symbol, trade_qty, side="buy")
    st.sidebar.success(message)
if col2.button("Sell Now"):
    message = place_paper_trade(trade_symbol, trade_qty, side="sell")
    st.sidebar.success(message)

# --- RL Weight Tracker Visualization ---
if st.sidebar.checkbox("Show Strategy Weights"):
    weights = load_model().get("weights", {})
    if weights:
        track_rl_weights(weights)
    else:
        st.info("No weights available yet. Run a backtest first.")

# --- Alert System (Log Only for Now) ---
st.sidebar.subheader("📣 Strategy Alert Preview")
alert_price = st.sidebar.number_input("Trigger Price", min_value=0.0, value=100.0, step=1.0)
alert_symbol = st.sidebar.text_input("Alert Symbol", "AAPL")
latest_price = 0.0

try:
    latest_price = api.get_stock_latest_trade(alert_symbol).price
    st.sidebar.write(f"🔎 Current Price: ${latest_price:.2f}")
    if latest_price >= alert_price:
        st.sidebar.success("📈 Alert: Price has reached or exceeded your target!")
    else:
        st.sidebar.info("Price has not yet reached the target.")
except Exception as e:
    st.sidebar.error(f"Failed to retrieve price: {e}")

# --- Deployment Note ---
st.markdown("""
---
🌐 This dashboard is now ready for **Streamlit Cloud Deployment**.
To deploy:
1. Push this file to a GitHub repo
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect the repo and deploy!
""")
