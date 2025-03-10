import yfinance as yf
import pandas as pd
import numpy as np
import talib

#  DOWNLOAD HOURLY S&P 500 DATA FOR 2 YEARS
ticker = "^GSPC"
data = yf.download(ticker, period="2y", interval="1h", auto_adjust=False)

#  REMOVE ROWS WITH MISSING VALUES AND CONVERT TO FLOAT
data = data.dropna().astype(float)

#  PREPROCESSING: COMPUTE CANDLE WICK SHAPES
data["Body_Size"] = abs(data["Close"] - data["Open"])  # Candle body size

# Convert OHLC columns to explicitly 1D arrays
open_arr = data["Open"].to_numpy().flatten()
high_arr = data["High"].to_numpy().flatten()
low_arr  = data["Low"].to_numpy().flatten()
close_arr = data["Close"].to_numpy().flatten()

# Compute elementwise maximum and minimum using NumPy
upper_max = np.maximum(open_arr, close_arr)
lower_min = np.minimum(open_arr, close_arr)

# Compute Upper and Lower Wicks and recreate Series with original index
upper_wick = pd.Series(high_arr - upper_max, index=data.index)
lower_wick = pd.Series(lower_min - low_arr, index=data.index)

data["Upper_Wick"] = upper_wick
data["Lower_Wick"] = lower_wick

data["Candle_Type"] = (data["Close"] > data["Open"]).astype(int)  # 1 = Bullish, 0 = Bearish

#  IDENTIFY CANDLESTICK PATTERNS
patterns = {
    "Doji": talib.CDLDOJI,
    "Hammer": talib.CDLHAMMER,
    "Inverted Hammer": talib.CDLINVERTEDHAMMER,
    "Shooting Star": talib.CDLSHOOTINGSTAR,
    "Engulfing": talib.CDLENGULFING,  # Handles both Bullish & Bearish
    "Morning Star": talib.CDLMORNINGSTAR,
    "Evening Star": talib.CDLEVENINGSTAR,
    "Dragonfly Doji": talib.CDLDRAGONFLYDOJI,
    "Gravestone Doji": talib.CDLGRAVESTONEDOJI,
    "Marubozu": talib.CDLMARUBOZU,
    "Spinning Top": talib.CDLSPINNINGTOP
}

# Use the flattened arrays when calling TA-Lib functions
for pattern, func in patterns.items():
    data[pattern] = func(open_arr, high_arr, low_arr, close_arr)

#  FILTER ROWS WITH DETECTED PATTERNS
candlestick_signals = data[(data.iloc[:, 6:] != 0).any(axis=1)]

#  SAVE THE PROCESSED DATASET
candlestick_signals.to_csv("sp500_candlestick_patterns_2y.csv")

print(candlestick_signals.head())


print("Dataset saved as 'sp500_candlestick_patterns_2y.csv'.")
