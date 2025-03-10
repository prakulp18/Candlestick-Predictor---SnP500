import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Load the Dataset ---
data = pd.read_csv("sp500_candlestick_patterns_2y.csv", index_col=0, parse_dates=True)

# --- 2. Ensure Patterns Are Numeric ---
# Convert all pattern columns to float
pattern_columns = ["Doji", "Hammer", "Inverted Hammer", "Shooting Star", "Engulfing",
                   "Morning Star", "Evening Star", "Dragonfly Doji", "Gravestone Doji",
                   "Marubozu", "Spinning Top"]

data[pattern_columns] = data[pattern_columns].apply(pd.to_numeric, errors='coerce')

# --- 3. Count Occurrences of Each Pattern ---
pattern_counts = (data[pattern_columns] != 0).sum()

# --- 4. Display the Count Data ---
print("\nCandlestick Pattern Counts:")
print(pattern_counts)

# --- 5. Visualize the Data ---
plt.figure(figsize=(12, 6))
pattern_counts.sort_values().plot(kind="barh", color="skyblue", edgecolor="black")
plt.xlabel("Number of Occurrences")
plt.ylabel("Candlestick Pattern")
plt.title("Number of Occurrences of Each Candlestick Pattern")
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.show()
