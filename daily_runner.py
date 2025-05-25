from strategy_dashboard import auto_daily_retrain, send_email_alert
from datetime import datetime

if __name__ == "__main__":
    log_file = auto_daily_retrain(symbol="AAPL", days=90)
    
    with open(log_file, "r") as f:
        summary = f.read()
    
    send_email_alert("📊 Daily Trading Strategy Summary", summary)
    
    # Optional: append to run log
    with open("run.log", "a") as f:
        f.write(f"[{datetime.now()}] Completed daily retrain. Log: {log_file}\n")
