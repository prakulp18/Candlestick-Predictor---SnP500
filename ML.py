import pandas as pd
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# --- 1️⃣ Load the dataset ---
data = pd.read_csv("sp500_candlestick_patterns_2y.csv", index_col=0, parse_dates=True)

# --- 2️⃣ Ensure OHLCV and Pattern Columns Are Numeric ---
pattern_columns = ["Doji", "Hammer", "Inverted Hammer", "Shooting Star", "Engulfing",
                   "Morning Star", "Evening Star", "Dragonfly Doji", "Gravestone Doji",
                   "Marubozu", "Spinning Top"]

feature_columns = ["Open", "High", "Low", "Close", "Volume", "Body_Size", "Upper_Wick", "Lower_Wick",
                   "RSI_14", "SMA_10", "SMA_50", "EMA_10", "EMA_50",
                   "Upper_Band", "Middle_Band", "Lower_Band",
                   "ATR_14", "MACD", "MACD_Signal", "ROC", "STDDEV_14"]

# Convert to numeric to ensure consistency
data[feature_columns + pattern_columns] = data[feature_columns + pattern_columns].apply(pd.to_numeric, errors='coerce')

# Drop rows where key features are missing
data = data.dropna(subset=feature_columns)

# --- 3️⃣ Remove Outliers Using Interquartile Range (IQR) ---
def remove_outliers(df, columns):
    Q1 = df[columns].quantile(0.25)
    Q3 = df[columns].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[~((df[columns] < lower_bound) | (df[columns] > upper_bound)).any(axis=1)]

data = remove_outliers(data, feature_columns)

# --- 4️⃣ Assign Pattern Labels ---
def assign_pattern(row):
    for pattern in pattern_columns:
        if row[pattern] != 0:
            return pattern
    return None  # Use None to ensure filtering consistency

data["Pattern"] = data.apply(assign_pattern, axis=1)

# Drop rows where no pattern was found
data = data.dropna(subset=["Pattern"])

# --- 5️⃣ Prepare Features (X) and Labels (y) ---
X = data[feature_columns]
y = data["Pattern"]

# Encode labels into numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Verify that X and y have the same number of rows
print(f"✅ Number of samples in X: {X.shape[0]}")
print(f"✅ Number of labels in y: {len(y_encoded)}")

# --- 6️⃣ Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Normalize input features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 7️⃣ Define the Hyperparameter Search Space ---
def build_model(hp):
    model = keras.Sequential()
    
    # Input layer
    model.add(keras.layers.Dense(hp.Int('units_input', min_value=64, max_value=256, step=64), 
                                 activation=hp.Choice('activation_input', ['relu', 'tanh']), 
                                 input_shape=(X_train_scaled.shape[1],)))
    
    # Hidden layers (number and size are tunable)
    for i in range(hp.Int('num_layers', 1, 4)):  # Between 1 to 4 hidden layers
        model.add(keras.layers.Dense(hp.Int(f'units_{i}', min_value=32, max_value=256, step=32),
                                     activation=hp.Choice('activation_hidden', ['relu', 'tanh'])))
        model.add(keras.layers.Dropout(hp.Float(f'dropout_{i}', min_value=0.2, max_value=0.5, step=0.1)))
    
    # Output layer
    model.add(keras.layers.Dense(len(label_encoder.classes_), activation='softmax'))
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', [0.0001, 0.001, 0.01])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# --- 8️⃣ Hyperparameter Optimization Using Keras Tuner ---
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=50,
    factor=3,
    directory='hyperband_tuning',
    project_name='candlestick_ann_optimization'
)

# Early stopping to prevent overfitting
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Run the hyperparameter search
tuner.search(X_train_scaled, y_train, epochs=50, validation_data=(X_test_scaled, y_test), 
             callbacks=[early_stopping])

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print("\nBest Hyperparameters Found:")
print(f"- Input Layer Neurons: {best_hps.get('units_input')}")
print(f"- Activation Function: {best_hps.get('activation_input')}")
print(f"- Number of Hidden Layers: {best_hps.get('num_layers')}")
for i in range(best_hps.get('num_layers')):
    print(f"  - Layer {i+1} Neurons: {best_hps.get(f'units_{i}')}")
    print(f"  - Dropout Rate: {best_hps.get(f'dropout_{i}')}")
print(f"- Learning Rate: {best_hps.get('learning_rate')}")

# --- 9️⃣ Train the Best Model ---
best_model = tuner.hypermodel.build(best_hps)

history = best_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, 
                         validation_data=(X_test_scaled, y_test), callbacks=[early_stopping])

# --- 🔟 Evaluate the Model ---
y_pred = np.argmax(best_model.predict(X_test_scaled), axis=1)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Optimized Model Accuracy: {accuracy:.2f}")

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# --- 🔟 Save the Optimized Model ---
best_model.save("optimized_candlestick_ann_model.h5")
import joblib
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("\nModel training complete. Optimized model saved as 'optimized_candlestick_ann_model.h5'.")
