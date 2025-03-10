import pandas as pd
import numpy as np
import talib

# Load the original dataset
data = pd.read_csv("sp500_candlestick_patterns_2y.csv", index_col=0, parse_dates=True)

# Ensure OHLC and Volume columns are numeric
numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Drop rows with missing values in OHLCV data
data = data.dropna(subset=numeric_columns)

# Add candlestick structural features
data["Body_Size"] = abs(data["Close"] - data["Open"])
data["Upper_Wick"] = data["High"] - data[["Open", "Close"]].max(axis=1)
data["Lower_Wick"] = data[["Open", "Close"]].min(axis=1) - data["Low"]
data["Candle_Type"] = np.where(data["Close"] > data["Open"], 1, 0)  # 1 for bullish, 0 for bearish

# Compute technical indicators
data["RSI_14"] = talib.RSI(data["Close"], timeperiod=14)
data["SMA_10"] = talib.SMA(data["Close"], timeperiod=10)
data["SMA_50"] = talib.SMA(data["Close"], timeperiod=50)
data["EMA_10"] = talib.EMA(data["Close"], timeperiod=10)
data["EMA_50"] = talib.EMA(data["Close"], timeperiod=50)

# Bollinger Bands
data["Upper_Band"], data["Middle_Band"], data["Lower_Band"] = talib.BBANDS(data["Close"], timeperiod=20)

# Volatility indicators
data["ATR_14"] = talib.ATR(data["High"], data["Low"], data["Close"], timeperiod=14)
data["STDDEV_14"] = talib.STDDEV(data["Close"], timeperiod=14)

# Trend indicators
data["MACD"], data["MACD_Signal"], _ = talib.MACD(data["Close"], fastperiod=12, slowperiod=26, signalperiod=9)
data["ROC"] = talib.ROCP(data["Close"], timeperiod=10)

# Drop rows with NaN values after feature generation
data = data.dropna()

# Overwrite the original file with the updated data
data.to_csv("sp500_candlestick_patterns_2y.csv")

# Display a preview of the dataset
print("\nFeature engineering complete. Updated dataset saved back to 'sp500_candlestick_patterns_2y.csv'.")
print("\nSample of the modified dataset:")
print(data.head())
