# Streamlit Strategy Dashboard

## 🚀 Features
- Strategy backtesting and comparison
- Reinforcement learning with rolling weights
- Live trade simulation via Alpaca API
- Email alerts and performance summary
- Daily retraining automation (via `daily_runner.py`)

## 🔧 Setup
1. Create a virtual environment (optional)
2. Install dependencies:
    pip install streamlit pandas matplotlib numpy alpaca-trade-api

3. Replace `your_api_key_here` and `your_secret_key_here` in `strategy_dashboard.py`
4. Replace email credentials in `send_email_alert()` function

## ▶️ Run Locally
    streamlit run strategy_dashboard.py

## ⏰ Automate Daily Retraining
Schedule `daily_runner.py` using Task Scheduler (Windows) or `cron` (Linux/Mac).

## 🌐 Deploy Online
1. Push to GitHub
2. Go to https://streamlit.io/cloud
3. Deploy your app by connecting the repo