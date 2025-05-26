import streamlit as st
from streamlit_autorefresh import st_autorefresh
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
import smtplib, ssl
from email.message import EmailMessage

# --- Auto-Refresh ---
st_autorefresh(interval=60000, key="auto_refresh")  # Refresh every 60 seconds

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
        api.submit_order(symbol=symbol, qty=qty, side=side, type='market', time_in_force='gtc')
        return f"✅ {side.upper()} order for {qty} shares of {symbol} placed."
    except Exception as e:
        return f"❌ Failed to place order: {e}"

def get_latest_price(symbol):
    try:
        trade = api.get_latest_trade(symbol)
        return trade.price
    except Exception as e:
        st.error(f"Failed to retrieve price: {e}")
        return None

def track_rl_weights(weights):
    df = pd.DataFrame.from_dict(weights, orient='index', columns=['Weight'])
    df = df.sort_values(by='Weight', ascending=False)
    st.subheader("📊 Reinforcement Learning Strategy Weights")
    st.bar_chart(df)

# --- Strategy Functions ---
def sma_crossover_strategy(data): return [random.choice(['buy', 'sell', 'hold']) for _ in data]
def buy_and_hold_strategy(data): return ['buy'] + ['hold'] * (len(data)-1)
def rsi_strategy(data): return [random.choice(['buy', 'sell', 'hold']) for _ in data]
def momentum_strategy(data): return [random.choice(['buy', 'sell', 'hold']) for _ in data]
def mean_reversion_strategy(data): return [random.choice(['buy', 'sell', 'hold']) for _ in data]
def macd_strategy(data): return [random.choice(['buy', 'sell', 'hold']) for _ in data]
def bollinger_strategy(data): return [random.choice(['buy', 'sell', 'hold']) for _ in data]
def ema_strategy(data): return [random.choice(['buy', 'sell', 'hold']) for _ in data]

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
        'sharpe_ratio': np.mean(portfolio_value)/np.std(portfolio_value),
        'max_drawdown': max(np.maximum.accumulate(portfolio_value) - portfolio_value),
        'win_rate': sum(1 for x in signals if x == 'buy') / len(signals),
        'num_trades': len(trades)
    }
    return portfolio_value, trades, metrics

def load_model(): return {"weights": {}}
def save_model(weights): pass
def reinforcement_learn(weights, metrics): return {k: random.random() for k in metrics}
def rolling_backtest(data, strategy_func): return [10000 + i * random.uniform(-10, 10) for i in range(10)]
def select_best_strategy(metrics): return max(metrics.items(), key=lambda x: x[1]['final_value'])[0]

# --- Auto Retrain ---
def auto_daily_retrain(symbol="AAPL", days=90):
    from pathlib import Path
    Path("logs").mkdir(exist_ok=True)
    bars = fetch_data(symbol, days)

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

# --- Streamlit UI ---
st.title("📈 AI-Powered Strategy Backtester")
symbol = st.sidebar.text_input("Symbol", "AAPL", key="symbol_input")
days = st.sidebar.slider("Days of Data", 30, 180, 90, key="days_slider")

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
selected = st.sidebar.multiselect("Select Strategies", strategies.keys(), default=list(strategies.keys()), key="strategy_select")

if st.sidebar.button("Run Backtest", key="run_backtest_btn"):
    bars = fetch_data(symbol, days)
    if not bars:
        st.warning("No data fetched.")
    else:
        metrics_all = {}
        full_trades = []

        for name in selected:
            st.subheader(f"Strategy: {name}")
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
            best = select_best_strategy(metrics_all)
            st.success(f"🏆 Best Strategy: {best}")
            updated = reinforcement_learn(load_model(), metrics_all)
            save_model({"weights": updated})
            track_rl_weights(updated)

            roll = rolling_backtest(bars, strategies[best])
            st.subheader("Rolling Backtest")
            fig2, ax2 = plt.subplots()
            ax2.plot(roll)
            ax2.set_title("Rolling Performance")
            st.pyplot(fig2)

            st.download_button(
                label="📥 Download Trades",
                data=pd.DataFrame(full_trades, columns=["Timestamp", "Action", "Shares", "Price"]).to_csv(index=False),
                file_name="trades.csv",
                mime="text/csv"
            )

# --- Email + Alerts ---
EMAIL_FROM = "reece.flew@gmail.com"
EMAIL_TO = "reece.flew@gmail.com"
EMAIL_PASS = "Hq1569801"
EMAIL_HOST = "smtp.gmail.com"
EMAIL_PORT = 465

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

# --- Unified Price Alert System ---
st.sidebar.subheader("📣 Real-Time Price Alert System")

alert_symbol = st.sidebar.text_input("Symbol to Watch", value="AAPL", key="alert_symbol_key")
target_price = st.sidebar.number_input("Target Price ($)", min_value=0.0, value=100.0, step=0.1, key="alert_price_key")

if st.sidebar.button("Activate Alert", key="activate_alert_btn"):
    st.session_state["active_alert"] = {
        "symbol": alert_symbol.upper(),
        "target": target_price
    }

if "active_alert" in st.session_state:
    alert = st.session_state["active_alert"]
    try:
        current_price = get_latest_price(alert["symbol"])
        if current_price:
            st.sidebar.write(f"🔎 {alert['symbol']} Current: ${current_price:.2f} | Target: ${alert['target']}")
            if current_price >= alert["target"]:
                st.sidebar.success(f"✅ {alert['symbol']} hit ${alert['target']}!")
                send_email_alert(
                    subject=f"{alert['symbol']} Alert Hit",
                    body=f"The stock reached ${current_price:.2f}, meeting your alert of ${alert['target']}."
                )
    except Exception as e:
        st.sidebar.error(f"Error checking price: {e}")
