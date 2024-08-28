import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Load the data
sp500 = yf.Ticker("^GSPC").history(period="max")

# Prepare the data by removing unnecessary columns and adding the 'Tomorrow' and 'Target' columns
sp500["Tomorrow"] = sp500["Close"].shift(-1)
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
sp500 = sp500.loc["1990-01-01":].copy()

# Define the predictors
predictors = ["Close", "Volume", "Open", "High", "Low"]

# Train the model with the historical data
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
train = sp500.iloc[:-1]  # Use all data except the last row for training
model.fit(train[predictors], train["Target"])

# Prepare the latest data point for prediction
latest_data = sp500.iloc[-1][predictors].values.reshape(1, -1)

# Predict the target for tomorrow
tomorrow_prediction = model.predict(latest_data)

# Create a Streamlit app
st.title("S&P 500 Prediction")

# Display the latest closing price
st.subheader("Latest S&P 500 Data")
st.write(sp500.iloc[-1][["Open", "High", "Low", "Close", "Volume"]])

# Plot the closing price history
st.subheader("Historical Closing Prices")
st.line_chart(sp500["Close"])

# Plot the volume history
st.subheader("Historical Trading Volume")
st.bar_chart(sp500["Volume"])

# Display the prediction
st.subheader("Prediction for Tomorrow")
if tomorrow_prediction[0] == 1:
    st.success("The S&P 500 is predicted to go up tomorrow.")
else:
    st.warning("The S&P 500 is predicted to go down tomorrow.")

# Optional: Show the historical data
if st.checkbox("Show Historical Data"):
    st.write(sp500.tail(10))
