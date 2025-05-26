import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from alpaca_trade_api.rest import REST, TimeFrame
from alpaca_trade_api.rest import TimeFrameUnit
import smtplib, ssl
from email.message import EmailMessage

# --- Auto-Refresh ---
st_autorefresh(interval=60000, key="auto_refresh")

# --- Config ---
API_KEY = st.secrets["alpaca_api_key"]
SECRET_KEY = st.secrets["alpaca_secret_key"]
BASE_URL = "https://paper-api.alpaca.markets"
api = REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

# --- Email Credentials from Secrets ---
EMAIL_FROM = st.secrets["email_from"]
EMAIL_TO = st.secrets["email_to"]
EMAIL_PASS = st.secrets["email_pass"]
EMAIL_HOST = st.secrets.get("email_host", "smtp.gmail.com")
EMAIL_PORT = st.secrets.get("email_port", 465)

# --- Utility Functions ---
def fetch_data(symbol, days=90):
    end = datetime.today()
    start = end - timedelta(days=days)
    try:
        bars = api.get_bars(symbol, TimeFrame(1, TimeFrameUnit.Day),
                            start=start.strftime("%Y-%m-%d"),
                            end=end.strftime("%Y-%m-%d"),
                            limit=days, feed='iex')
        return bars
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return []

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
        msg["From"] = EMAIL_FROM
        msg["To"] = EMAIL_TO
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(EMAIL_HOST, EMAIL_PORT, context=context) as server:
            server.login(EMAIL_FROM, EMAIL_PASS)
            server.send_message(msg)
    except Exception as e:
        st.sidebar.error(f"📧 Email failed: {e}")

def track_rl_weights(weights):
    df = pd.DataFrame.from_dict(weights, orient='index', columns=['Weight'])
    df = df.sort_values(by='Weight', ascending=False)
    st.subheader("📊 Strategy Weights")
    st.bar_chart(df)

# --- Strategies ---
def random_strategy(data): return [random.choice(['buy', 'sell', 'hold']) for _ in data]
def buy_and_hold_strategy(data): return ['buy'] + ['hold'] * (len(data) - 1)

# --- Backtesting ---
def backtest_strategy(data, signals):
    portfolio_value = [10000]
    trades = []
    for i, signal in enumerate(signals):
        if signal == 'buy':
            portfolio_value.append(portfolio_value[-1] * 1.01)
            trades.append((i, 'BUY', 10, 100 + i))
        elif signal == 'sell':
            portfolio_value.append(portfolio_value[-1] * 0.99)
            trades.append((i, 'SELL', 10, 100 + i))
        else:
            portfolio_value.append(portfolio_value[-1])
    metrics = {
        'final_value': portfolio_value[-1],
        'sharpe_ratio': np.mean(portfolio_value) / np.std(portfolio_value),
        'max_drawdown': max(np.maximum.accumulate(portfolio_value) - portfolio_value),
        'win_rate': sum(1 for x in signals if x == 'buy') / len(signals),
        'num_trades': len(trades)
    }
    return portfolio_value, trades, metrics

# --- Streamlit UI ---
st.title("📈 AI Trading Dashboard")
symbol = st.sidebar.text_input("Symbol", "AAPL")
days = st.sidebar.slider("Days of Data", 30, 180, 90)

strategies = {
    "Random": random_strategy,
    "Buy & Hold": buy_and_hold_strategy
}
selected = st.sidebar.multiselect("Select Strategies", list(strategies.keys()), default=list(strategies.keys()))

if st.sidebar.button("Run Backtest"):
    bars = fetch_data(symbol, days)
    if not bars:
        st.warning("No data fetched.")
    else:
        metrics_all = {}
        full_trades = []

        for name in selected:
            st.subheader(name)
            sig = strategies[name](bars)
            perf, trades, metrics = backtest_strategy(bars, sig)
            metrics_all[name] = metrics
            full_trades.extend(trades)

            st.write(metrics)
            fig, ax = plt.subplots()
            ax.plot(perf)
            ax.set_title(f"{name} Portfolio Value")
            st.pyplot(fig)

        if metrics_all:
            best = max(metrics_all.items(), key=lambda x: x[1]['final_value'])[0]
            st.success(f"🏆 Best Strategy: {best}")
            updated_weights = {k: random.random() for k in metrics_all}
            track_rl_weights(updated_weights)

            st.download_button(
                "📥 Download Trades",
                data=pd.DataFrame(full_trades, columns=["Timestamp", "Action", "Shares", "Price"]).to_csv(index=False),
                file_name="trades.csv",
                mime="text/csv"
            )

# --- Price Alert ---
st.sidebar.subheader("📣 Price Alert System")
alert_symbol = st.sidebar.text_input("Watch Symbol", "AAPL", key="alert_symbol")
target_price = st.sidebar.number_input("Target Price", min_value=0.0, value=100.0, step=0.1, key="alert_price")

if st.sidebar.button("Activate Alert", key="activate_alert"):
    st.session_state["active_alert"] = {"symbol": alert_symbol.upper(), "target": target_price}

if "active_alert" in st.session_state:
    alert = st.session_state["active_alert"]
    current_price = get_latest_price(alert["symbol"])
    if current_price:
        st.sidebar.write(f"🔎 {alert['symbol']} @ ${current_price:.2f} (Target: ${alert['target']})")
        if current_price >= alert["target"]:
            st.sidebar.success(f"✅ {alert['symbol']} hit ${alert['target']}!")
            send_email_alert(
                subject=f"{alert['symbol']} Alert Hit",
                body=f"{alert['symbol']} reached ${current_price:.2f}, your target of ${alert['target']}."
            )
