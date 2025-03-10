# Candlestick Pattern Classification with ANN

This project implements a deep learning model for recognizing and classifying candlestick patterns in S&P 500 stock market data. The model is built using an Artificial Neural Network (ANN) and is optimized through hyperparameter tuning with Keras Tuner.

## Project Overview
The pipeline follows these key steps:
1. **Data Collection:** Extracting hourly S&P 500 data for the last two years using `yfinance`.
2. **Feature Engineering:** Computing technical indicators and candlestick features using `TA-Lib`.
3. **Pattern Labeling:** Identifying candlestick patterns and assigning labels.
4. **Data Preprocessing:** Cleaning the dataset, normalizing features, and encoding labels.
5. **Model Training:** Using a fully connected ANN with optimized hyperparameters.
6. **Evaluation:** Measuring classification accuracy and generating reports.
7. **Visualization:** Plotting occurrences of candlestick patterns.

## File Descriptions

- `S&P500_candlestick.py` - Downloads S&P 500 data, detects candlestick patterns, and saves the dataset.
- `feature_enhancer.py` - Computes technical indicators and additional features.
- `ML.py` - Prepares data, trains an ANN classifier, and optimizes the model using Keras Tuner.
- `plotter_snp500.py` - Generates visualizations of candlestick pattern occurrences.
- `candlestick_ann_model.h5` - Trained ANN model for candlestick classification.
- `optimized_candlestick_ann_model.h5` - Optimized ANN model with hyperparameter tuning.
- `candlestick_classifier.pkl` - Saved classifier model.
- `label_encoder.pkl` - Label encoder for mapping patterns to numerical values.
- `scaler.pkl` - MinMax scaler used for feature normalization.
- `sp500_candlestick_patterns_2y.csv` - Processed dataset containing candlestick patterns.

## Installation & Dependencies
To run this project, install the required dependencies:

```bash
pip install pandas numpy tensorflow keras keras-tuner scikit-learn talib matplotlib yfinance
```

## Usage
1. **Download and preprocess data**
   ```bash
   python S&P500_candlestick.py
   ```
2. **Enhance features with technical indicators**
   ```bash
   python feature_enhancer.py
   ```
3. **Train and optimize the ANN model**
   ```bash
   python ML.py
   ```
4. **Visualize pattern occurrences**
   ```bash
   python plotter_snp500.py
   ```

## Results
The final ANN model achieves high accuracy in classifying candlestick patterns. The visualization tool provides insights into pattern occurrences over time.

## Future Improvements
- Implement LSTM models for time-series forecasting.
- Integrate reinforcement learning for trading decisions.
- Extend analysis to other financial markets.

## Author
This project belongs to Prakul Pandit and was developed for analyzing financial markets using AI-driven candlestick pattern recognition.

