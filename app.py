"""Streamlit application for LSTM-based stock price forecasting."""

from __future__ import annotations

from datetime import date, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, LSTM


st.set_page_config(page_title="Stock Market Prediction", page_icon="📈", layout="wide")


@st.cache_data(show_spinner=False)
def download_prices(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Download and normalize daily OHLCV data for a ticker."""
    data = yf.download(
        ticker,
        start=start,
        end=end + timedelta(days=1),
        auto_adjust=False,
        progress=False,
    )
    if data.empty:
        raise ValueError(f"No market data was returned for {ticker}.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Downloaded data is missing columns: {', '.join(sorted(missing))}.")
    return data.loc[:, sorted(required)].dropna().sort_index()


def make_sequences(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Create rolling LSTM inputs and one-step-ahead targets."""
    if len(values) <= window:
        raise ValueError(f"At least {window + 1} observations are required.")
    x = np.array([values[i - window : i] for i in range(window, len(values))])
    y = np.array([values[i] for i in range(window, len(values))])
    return x.reshape(-1, window, 1), y


def build_model(window: int) -> Sequential:
    model = Sequential(
        [
            LSTM(64, return_sequences=True, input_shape=(window, 1)),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def train_forecaster(
    prices: pd.Series, window: int, epochs: int, batch_size: int
) -> tuple[Sequential, MinMaxScaler, pd.DataFrame, float]:
    """Train on the chronological training portion and evaluate on the test portion."""
    values = prices.to_numpy(dtype="float32").reshape(-1, 1)
    split_index = int(len(values) * 0.8)
    if split_index <= window or len(values) - split_index < 2:
        raise ValueError("Choose a longer date range or a smaller look-back window.")

    scaler = MinMaxScaler()
    scaler.fit(values[:split_index])
    scaled = scaler.transform(values)
    x, y = make_sequences(scaled, window)
    target_indices = np.arange(window, len(values))
    train_mask = target_indices < split_index
    test_mask = ~train_mask

    model = build_model(window)
    early_stopping = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    model.fit(
        x[train_mask],
        y[train_mask],
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=0,
    )

    predicted = scaler.inverse_transform(model.predict(x[test_mask], verbose=0)).ravel()
    actual = scaler.inverse_transform(y[test_mask].reshape(-1, 1)).ravel()
    metrics = pd.DataFrame(
        {
            "Actual": actual,
            "Predicted": predicted,
        },
        index=prices.index[target_indices[test_mask]],
    )
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    return model, scaler, metrics, rmse


def forecast_next_days(
    model: Sequential, scaler: MinMaxScaler, prices: pd.Series, window: int, days: int
) -> pd.DataFrame:
    """Recursively forecast future closes from the latest look-back window."""
    recent = scaler.transform(prices.to_numpy(dtype="float32").reshape(-1, 1)).ravel().tolist()
    predictions: list[float] = []
    for _ in range(days):
        model_input = np.array(recent[-window:]).reshape(1, window, 1)
        next_scaled = float(model.predict(model_input, verbose=0)[0, 0])
        recent.append(next_scaled)
        predictions.append(float(scaler.inverse_transform([[next_scaled]])[0, 0]))
    future_dates = pd.bdate_range(prices.index[-1] + pd.Timedelta(days=1), periods=days)
    return pd.DataFrame({"Predicted Close": predictions}, index=future_dates)


st.title("📈 Stock Market Prediction with LSTM")
st.caption("Educational forecasting tool — predictions are estimates, not financial advice.")

with st.sidebar:
    st.header("Model settings")
    ticker = st.text_input("Ticker symbol", "AAPL").strip().upper()
    start = st.date_input("Start date", date.today() - timedelta(days=5 * 365))
    end = st.date_input("End date", date.today())
    window = st.slider("Look-back window (trading days)", 20, 120, 60, 5)
    epochs = st.slider("Maximum training epochs", 5, 100, 30, 5)
    batch_size = st.selectbox("Batch size", [16, 32, 64], index=1)
    forecast_days = st.slider("Forecast horizon (business days)", 1, 30, 5)
    train_button = st.button("Train model", type="primary", use_container_width=True)

if start >= end:
    st.error("The start date must be earlier than the end date.")
    st.stop()

try:
    data = download_prices(ticker, start, end)
except (ValueError, KeyError, TypeError, RuntimeError) as error:
    st.error(f"Could not load data: {error}")
    st.stop()

st.subheader(f"{ticker} historical prices")
st.line_chart(data["Close"], y_label="Close price")
st.dataframe(data.tail(10), use_container_width=True)

if train_button:
    with st.spinner("Training the LSTM model..."):
        try:
            model, scaler, evaluation, rmse = train_forecaster(
                data["Close"], window, epochs, batch_size
            )
            future = forecast_next_days(model, scaler, data["Close"], window, forecast_days)
        except (ValueError, KeyError, TypeError, RuntimeError) as error:
            st.error(f"Training failed: {error}")
            st.stop()

    latest_close = float(data["Close"].iloc[-1])
    next_close = float(future["Predicted Close"].iloc[0])
    movement = "UP 📈" if next_close >= latest_close else "DOWN 📉"
    change = (next_close - latest_close) / latest_close * 100

    metric_columns = st.columns(4)
    metric_columns[0].metric("Last close", f"${latest_close:,.2f}")
    metric_columns[1].metric("Next predicted close", f"${next_close:,.2f}", f"{change:+.2f}%")
    metric_columns[2].metric("Predicted movement", movement)
    metric_columns[3].metric("Test RMSE", f"${rmse:,.2f}")

    st.subheader("Test-set predictions")
    figure, axis = plt.subplots(figsize=(12, 4))
    axis.plot(evaluation.index, evaluation["Actual"], label="Actual", color="#1f77b4")
    axis.plot(evaluation.index, evaluation["Predicted"], label="Predicted", color="#ff7f0e")
    axis.set_ylabel("Close price")
    axis.set_xlabel("Date")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)

    st.subheader(f"Next {forecast_days} business-day forecast")
    st.line_chart(future, y_label="Predicted close")
    st.dataframe(future.style.format("${:,.2f}"), use_container_width=True)
    st.info(
        f"Mean absolute error on the held-out test period: "
        f"${mean_absolute_error(evaluation['Actual'], evaluation['Predicted']):,.2f}."
    )