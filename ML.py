import pandas as pd
import numpy as np
import yfinance as yf
import talib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# ---  Load the dataset ---
data = pd.read_csv("sp500_candlestick_patterns_2y.csv", index_col=0, parse_dates=True)

# ---  Ensure OHLC and pattern columns are numeric ---
pattern_columns = ["Doji", "Hammer", "Inverted Hammer", "Shooting Star", "Engulfing",
                   "Morning Star", "Evening Star", "Dragonfly Doji", "Gravestone Doji",
                   "Marubozu", "Spinning Top"]

ohlc_columns = ["Open", "High", "Low", "Close"]

# Convert all relevant columns to numeric
data[pattern_columns + ohlc_columns] = data[pattern_columns + ohlc_columns].apply(pd.to_numeric, errors='coerce')

# Drop rows with missing values
data = data.dropna()

# ---  Create Labels (y) for the Model ---
# Assign pattern name based on detected pattern (only 1 per row)
def assign_pattern(row):
    for pattern in pattern_columns:
        if row[pattern] != 0:
            return pattern  # Return the first detected pattern
    return "None"

data["Pattern"] = data.apply(assign_pattern, axis=1)

# Drop "None" rows where no pattern was detected
data = data[data["Pattern"] != "None"]

# --- 4️⃣ Prepare Features (X) and Labels (y) ---
X = data[ohlc_columns]  # Features: Open, High, Low, Close
y = data["Pattern"]     # Label: Candlestick pattern

# Encode labels into numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Standardize features (important for some ML models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---  Train a Random Forest Classifier ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# ---  Evaluate the Model ---
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy:.2f}")

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# ---  Save the Model for Future Predictions ---
import joblib
joblib.dump(model, "candlestick_classifier.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("\n Model saved")
