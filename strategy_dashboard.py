# --- Strategy Functions ---
def sma_crossover_strategy(data):
    df = data.df.copy()
    df["sma_short"] = df["close"].rolling(window=10).mean()
    df["sma_long"] = df["close"].rolling(window=30).mean()
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

def buy_and_hold_strategy(data):
    return ['buy'] + ['hold'] * (len(data.df) - 1)

def rsi_strategy(data, period=14):
    df = data.df.copy()
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    df["rsi"] = rsi
    signals = []
    for r in rsi:
        if pd.isna(r):
            signals.append("hold")
        elif r < 30:
            signals.append("buy")
        elif r > 70:
            signals.append("sell")
        else:
            signals.append("hold")
    return signals

def momentum_strategy(data, period=10):
    df = data.df.copy()
    df["momentum"] = df["close"].diff(periods=period)
    signals = []
    for m in df["momentum"]:
        if pd.isna(m):
            signals.append("hold")
        elif m > 0:
            signals.append("buy")
        elif m < 0:
            signals.append("sell")
        else:
            signals.append("hold")
    return signals

def mean_reversion_strategy(data, window=20):
    df = data.df.copy()
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
