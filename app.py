# Streamlit app for NIFTY LSTM Trend Prediction (Demo Forecast to 2030)

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
from nsepy import get_history
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# App title
st.title("📈 NIFTY Daily Trend Forecast (2025–2030 LSTM Demo)")

# Load model
model = load_model("nifty_lstm_model.h5")

# Load and preprocess data
def load_data():
    data = get_history(symbol="NIFTY", index=True, index_symbol="NIFTY 50",
                       start=date(2019,1,1), end=date(2024,12,31))
    data = data[['Close']]
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    data.dropna(inplace=True)
    return data

# Create sequences
def create_sequences(data, seq_length=60):
    X = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
    return np.array(X)

# Predict future trends
def forecast_future_trend(data, model, steps=1250):  # Approx. 5 years (250 days/year)
    predictions = []
    last_sequence = data[-60:]
    for _ in range(steps):
        input_seq = last_sequence.reshape(1, 60, 1)
        pred = model.predict(input_seq)[0][0]
        predictions.append(pred)
        last_value = last_sequence[-1][0]
        change = last_value * 0.002 if pred > 0.5 else -last_value * 0.002
        new_value = last_value + change
        new_point = np.array([[new_value]])
        last_sequence = np.append(last_sequence[1:], new_point, axis=0)
    return predictions

# Main app
st.markdown("This app predicts the next day's **Up/Down trend** of the NIFTY 50 index using an LSTM deep learning model, and simulates future trends up to 2030.")

# Load and prepare data
data = load_data()
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['Close']])
X = create_sequences(scaled_data)

# Take the latest sequence for prediction
last_sequence = X[-1].reshape(1, 60, 1)

# Predict next day trend
prediction = model.predict(last_sequence)
predicted_class = int(prediction[0][0] > 0.5)

# Output
trend = "⬆️ UP" if predicted_class == 1 else "⬇️ DOWN"
st.metric(label="Predicted Trend for Next Day", value=trend)

# Simulate next 5 years
st.subheader("🔮 Forecast: NIFTY Trend Simulation (2025–2030)")
future_preds = forecast_future_trend(scaled_data, model, steps=1250)
pred_labels = [1 if p > 0.5 else 0 for p in future_preds]

plt.figure(figsize=(12, 4))
plt.plot(pred_labels, color='green' if sum(pred_labels)/len(pred_labels) > 0.5 else 'red')
plt.title("Predicted Up/Down Trend over Next 5 Years (Demo)")
plt.xlabel("Days")
plt.ylabel("Trend (1=Up, 0=Down)")
st.pyplot(plt)

# Plot historical closing
st.subheader("📊 Historical NIFTY Closing Price")
plt.figure(figsize=(10, 4))
plt.plot(data['Close'].values[-100:])
plt.title("NIFTY Last 100 Days")
plt.xlabel("Days")
plt.ylabel("Closing Price")
st.pyplot(plt)

st.caption("Demo model trained on data till 2024; forecasts simulated using directional trends")
