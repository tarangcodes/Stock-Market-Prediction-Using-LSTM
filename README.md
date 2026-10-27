# Stock Market Prediction Using LSTM

An educational stock forecasting application using a Long Short-Term Memory
(LSTM) neural network. Historical daily OHLCV data is downloaded from
Yahoo Finance, and the Streamlit app predicts future closing prices and the
next expected price movement.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m streamlit run app.py
```

Open the local Streamlit URL shown in the terminal, usually
`http://localhost:8501`.

The notebook in `model.ipynb` contains the exploratory analysis, chronological
train/test split, leakage-safe scaling, LSTM training, evaluation, and
multi-day recursive forecasting workflow.

> Forecasts are estimates for educational purposes and are not financial advice.
