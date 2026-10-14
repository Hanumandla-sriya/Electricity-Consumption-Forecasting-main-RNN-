"""
Electricity Consumption Forecasting — Streamlit Dashboard
============================================================
Loads the trained LSTM model + scaler from preprocessing.ipynb /
lstm_model.ipynb and serves an interactive dashboard with:

  - Live electricity consumption graph
  - Forecast chart with selectable horizon
  - Daily average consumption
  - Custom data upload
  - Actual vs predicted comparison (backtest on the held-out test set)

Run with:
    streamlit run app.py

Expected project layout (paths below are relative to this file):
    models/best_model.pth
    models/scaler.pkl
    notebook/household_power_consumption.txt
    outputs/predictions.csv
"""

import os

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pth")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "notebook", "household_power_consumption.txt")
PREDICTIONS_PATH = os.path.join(BASE_DIR, "outputs", "predictions.csv")

SEQ_LENGTH = 60
TARGET_COL = "Global_active_power"
FEATURE_COLS = [
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
    "Hour",
    "Day",
    "Month",
    "Year",
    "Weekday",
]

HORIZON_OPTIONS = {
    "Next 15 minutes": 15,
    "Next 1 hour": 60,
    "Next 6 hours": 360,
    "Next 24 hours": 1440,
}

st.set_page_config(
    page_title="Electricity Consumption Forecasting",
    page_icon="⚡",
    layout="wide",
)

# ----------------------------------------------------------------------
# Model definition — must match lstm_model.ipynb exactly to load weights
# ----------------------------------------------------------------------

class LSTMForecaster(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size=128,
                 num_layers=4,
                 dropout=0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)

        # Last timestep output
        out = out[:, -1, :]

        out = self.fc(out)

        return out.squeeze(-1)

# ----------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model and scaler...")
def load_artifacts():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, None, device

    scaler = joblib.load(SCALER_PATH)

    model = LSTMForecaster(input_size=len(FEATURE_COLS))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    return model, scaler, device


@st.cache_data(show_spinner="Loading dataset (this can take a moment)...")
def load_default_data():
    if not os.path.exists(DEFAULT_DATA_PATH):
        return None
    df = pd.read_csv(
        DEFAULT_DATA_PATH,
        sep=";",
        na_values="?",
        low_memory=False,
    )
    return preprocess_raw(df)


@st.cache_data(show_spinner="Loading backtest predictions...")
def load_predictions():
    if not os.path.exists(PREDICTIONS_PATH):
        return None
    preds = pd.read_csv(PREDICTIONS_PATH, parse_dates=["Datetime"])
    return preds


def preprocess_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Same cleaning + feature engineering pipeline as preprocessing.ipynb."""
    df = df.copy()

    numeric_cols = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Datetime" not in df.columns and {"Date", "Time"}.issubset(df.columns):
        df["Datetime"] = pd.to_datetime(
            df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S"
        )
        df.drop(columns=["Date", "Time"], inplace=True)
    elif "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"])

    df.set_index("Datetime", inplace=True)
    df.sort_index(inplace=True)
    df.ffill(inplace=True)
    df.dropna(inplace=True)

    df["Hour"] = df.index.hour
    df["Day"] = df.index.day
    df["Month"] = df.index.month
    df["Year"] = df.index.year
    df["Weekday"] = df.index.dayofweek

    return df


# ----------------------------------------------------------------------
# Forecasting
# ----------------------------------------------------------------------
def recursive_forecast(df: pd.DataFrame, model, scaler, device, n_steps: int) -> pd.DataFrame:
    """
    Roll the model forward n_steps beyond the end of df.

    The model only predicts Global_active_power. For future exogenous
    features (Voltage, Sub_metering, etc.) we don't have ground truth,
    so we carry forward the last known values (naive persistence) —
    a reasonable simplification for a portfolio project, but worth
    stating plainly: this is not a true multivariate forecast of every
    feature, only of the target.
    """
    history = df[FEATURE_COLS].tail(SEQ_LENGTH).copy()
    last_known_exog = history.iloc[-1][
        [c for c in FEATURE_COLS if c not in ("Hour", "Day", "Month", "Year", "Weekday")]
    ]

    scaled_window = scaler.transform(history.values)
    window = torch.tensor(scaled_window, dtype=torch.float32).unsqueeze(0).to(device)

    last_timestamp = df.index[-1]
    preds = []
    timestamps = []

    model.eval()
    with torch.no_grad():
        for step in range(1, n_steps + 1):
            pred_scaled_input = window
            pred = model(pred_scaled_input).item()

            next_ts = last_timestamp + pd.Timedelta(minutes=step)
            timestamps.append(next_ts)
            preds.append(pred)

            next_row = last_known_exog.copy()
            next_row["Hour"] = next_ts.hour
            next_row["Day"] = next_ts.day
            next_row["Month"] = next_ts.month
            next_row["Year"] = next_ts.year
            next_row["Weekday"] = next_ts.dayofweek
            next_row = next_row.reindex(FEATURE_COLS)

            next_scaled = scaler.transform(next_row.values.reshape(1, -1))
            next_scaled_tensor = torch.tensor(next_scaled, dtype=torch.float32).to(device)

            window = torch.cat(
                [window[:, 1:, :], next_scaled_tensor.unsqueeze(0)], dim=1
            )

    return pd.DataFrame({"Datetime": timestamps, "Forecast": preds})


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("⚡ Electricity Consumption Forecasting")
st.caption("LSTM-based forecasting dashboard for household power consumption")

model, scaler, device = load_artifacts()

if model is None:
    st.error(
        "Could not find `models/best_model.pth` or `models/scaler.pkl`. "
        "Make sure app.py sits at the project root, alongside the `models/` folder."
    )
    st.stop()

# --- Sidebar: data source + horizon ---
st.sidebar.header("Settings")

if os.path.exists(DEFAULT_DATA_PATH):
    data_source = st.sidebar.radio(
        "Data source",
        ["Use bundled dataset", "Upload custom data"],
    )
else:
    st.sidebar.warning("Bundled dataset not found. Please upload your dataset.")
    data_source = "Upload custom data"

uploaded_file = None
if data_source == "Upload custom data":
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV (same format as household_power_consumption.txt: "
        "semicolon-separated, with Date/Time or Datetime column)",
        type=["csv", "txt"],
    )
    st.sidebar.caption(
        "Expected columns: Date, Time (or Datetime), Global_active_power, "
        "Global_reactive_power, Voltage, Global_intensity, Sub_metering_1/2/3"
    )

horizon_label = st.sidebar.selectbox("Forecast horizon", list(HORIZON_OPTIONS.keys()))
horizon_steps = HORIZON_OPTIONS[horizon_label]

st.sidebar.info(
    "⚠️ Forecasts beyond the target variable assume other readings "
    "(voltage, sub-metering, etc.) stay at their last known values. "
    "Only the target (Global_active_power) is truly forecasted."
)

# --- Load data based on source ---
if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file, sep=";", na_values="?", low_memory=False)
        data_df = preprocess_raw(raw_df)
        st.sidebar.success(f"Loaded {len(data_df):,} rows from uploaded file.")
    except Exception as e:
        st.sidebar.error(f"Could not process uploaded file: {e}")
        data_df = None
elif data_source == "Use bundled dataset":
    data_df = load_default_data()
    
else:
    data_df = None

if data_df is None or len(data_df) < SEQ_LENGTH:
    st.warning(f"Need at least {SEQ_LENGTH} rows of data to proceed.")
    st.stop()

# --- Tabs ---
tab_live, tab_forecast, tab_daily, tab_voltage, tab_cost, tab_backtest = st.tabs(
    [
        "📈 Live Consumption",
        "🔮 Forecast",
        "📊 Daily Average",
        "🔌 Voltage",
        "💰 Cost Estimator",
        "📉 Actual vs Predicted",
    ]
)

# ===== Tab 1: Live consumption =====
with tab_live:
    st.subheader("Recent Electricity Consumption")

    lookback_minutes = st.slider(
        "Show last N minutes", min_value=60, max_value=10080, value=1440, step=60
    )
    recent = data_df.tail(lookback_minutes)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recent.index,
            y=recent[TARGET_COL],
            mode="lines",
            name="Global Active Power (kW)",
            line=dict(color="#4C72B0"),
        )
    )
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="kW",
        height=450,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Current reading", f"{recent[TARGET_COL].iloc[-1]:.3f} kW")
    col2.metric("Average (window)", f"{recent[TARGET_COL].mean():.3f} kW")
    col3.metric("Peak (window)", f"{recent[TARGET_COL].max():.3f} kW")

# ===== Tab 2: Forecast =====
with tab_forecast:
    st.subheader(f"Forecast — {horizon_label}")

    if st.button("Generate Forecast", type="primary"):
        with st.spinner("Running recursive forecast..."):
            forecast_df = recursive_forecast(
                data_df, model, scaler, device, n_steps=horizon_steps
            )
        st.session_state["forecast_df"] = forecast_df

    if "forecast_df" in st.session_state:
        forecast_df = st.session_state["forecast_df"]
        history_tail = data_df[TARGET_COL].tail(180)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=history_tail.index,
                y=history_tail.values,
                mode="lines",
                name="Recent Actual",
                line=dict(color="#4C72B0"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_df["Datetime"],
                y=forecast_df["Forecast"],
                mode="lines",
                name="Forecast",
                line=dict(color="#DD8452", dash="dash"),
            )
        )
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="kW",
            height=450,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        peak_row = forecast_df.loc[forecast_df["Forecast"].idxmax()]
        st.info(
            f"⚠️ Highest predicted usage in this window: "
            f"**{peak_row['Forecast']:.3f} kW** at "
            f"**{peak_row['Datetime'].strftime('%Y-%m-%d %H:%M')}**"
        )

        with st.expander("View forecast data"):
            st.dataframe(forecast_df, use_container_width=True)
    else:
        st.caption("Click **Generate Forecast** to run the model.")

# ===== Tab 3: Daily average =====
with tab_daily:
    st.subheader("Daily Average Consumption")

    daily = data_df[TARGET_COL].resample("D").mean().dropna()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily.index,
            y=daily.values,
            mode="lines+markers",
            name="Daily Avg (kW)",
            line=dict(color="#55A868"),
        )
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Average kW",
        height=450,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Weekly Average**")
        weekly = data_df[TARGET_COL].resample("W").mean().dropna()
        st.line_chart(weekly)
    with col2:
        st.markdown("**Monthly Average**")
        monthly = data_df[TARGET_COL].resample("ME").mean().dropna()
        st.line_chart(monthly)

# ===== Tab: Voltage =====
with tab_voltage:
    st.subheader("Voltage Consumption")

    available_dates = sorted(data_df.index.normalize().unique())
    selected_date = st.selectbox(
        "Select a day to inspect",
        available_dates,
        index=len(available_dates) - 1,
        format_func=lambda d: d.strftime("%Y-%m-%d"),
    )

    day_data = data_df[data_df.index.normalize() == selected_date]

    if day_data.empty:
        st.warning("No data available for the selected day.")
    else:
        fig_v = go.Figure()
        fig_v.add_trace(
            go.Scatter(
                x=day_data.index,
                y=day_data["Voltage"],
                mode="lines",
                name="Voltage (V)",
                line=dict(color="#C44E52"),
            )
        )
        fig_v.update_layout(
            xaxis_title="Time",
            yaxis_title="Voltage (V)",
            height=400,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_v, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Average Voltage", f"{day_data['Voltage'].mean():.2f} V")
        col2.metric("Min Voltage", f"{day_data['Voltage'].min():.2f} V")
        col3.metric("Max Voltage", f"{day_data['Voltage'].max():.2f} V")

    st.markdown("**Daily Average Voltage (all days)**")
    daily_voltage = data_df["Voltage"].resample("D").mean().dropna()
    st.line_chart(daily_voltage)

# ===== Tab: Cost Estimator =====
with tab_cost:
    st.subheader("Electricity Cost Estimator")

    rate = st.number_input(
        "Price per unit (₹ per kWh)", min_value=0.0, value=8.0, step=0.5
    )

    st.markdown("### Cost for a date range (historical data)")
    min_date = data_df.index.min().date()
    max_date = data_df.index.max().date()

    date_range = st.date_input(
        "Select date range",
        value=(max_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        range_data = data_df.loc[str(start_date):str(end_date)]

        if range_data.empty:
            st.warning("No data in the selected range.")
        else:
            # Readings are per-minute kW; energy(kWh) = power(kW) * time(hours)
            total_kwh = range_data[TARGET_COL].sum() * (1 / 60)
            total_cost = total_kwh * rate

            col1, col2 = st.columns(2)
            col1.metric("Total Consumption", f"{total_kwh:.2f} kWh")
            col2.metric("Estimated Cost", f"₹{total_cost:,.2f}")
    else:
        st.info("Select a start and end date to see the cost.")

    st.markdown("---")
    st.markdown("### Predict cost for a specific date")

    last_data_date = data_df.index.max().date()
    target_date = st.date_input(
        "Pick a date to predict",
        value=last_data_date + pd.Timedelta(days=1),
        min_value=min_date,
        key="cost_predict_date",
    )

    if st.button("Predict Cost for This Date"):
        if target_date <= last_data_date:
            # Historical date — use actual readings, no model needed
            day_data = data_df[data_df.index.normalize() == pd.Timestamp(target_date)]
            if day_data.empty:
                st.warning("No data available for this historical date.")
            else:
                kwh = day_data[TARGET_COL].sum() * (1 / 60)
                cost = kwh * rate
                st.success(
                    f"**{target_date}** (actual data): "
                    f"{kwh:.2f} kWh → estimated cost **₹{cost:,.2f}**"
                )
        else:
            # Future date — recursively forecast forward and sum predicted kWh for that day
            day_start = pd.Timestamp(target_date)
            day_end = day_start + pd.Timedelta(days=1)
            minutes_ahead = int((day_end - data_df.index.max()).total_seconds() // 60)

            if minutes_ahead > 43200:  # ~30 days — recursive error compounds badly beyond this
                st.error(
                    "That date is too far ahead — recursive forecasting error compounds "
                    "heavily beyond ~30 days. Pick a nearer date."
                )
            else:
                with st.spinner(f"Forecasting up to {target_date}..."):
                    forecast_df = recursive_forecast(
                        data_df, model, scaler, device, n_steps=minutes_ahead
                    )
                day_forecast = forecast_df[
                    (forecast_df["Datetime"] >= day_start)
                    & (forecast_df["Datetime"] < day_end)
                ]
                if day_forecast.empty:
                    st.warning("No forecasted points fall within that date.")
                else:
                    kwh = day_forecast["Forecast"].sum() * (1 / 60)
                    cost = kwh * rate
                    st.success(
                        f"**{target_date}** (predicted): "
                        f"{kwh:.2f} kWh → estimated cost **₹{cost:,.2f}**"
                    )
                    st.caption(
                        "⚠️ Long-range forecasts use recursive prediction and carry "
                        "forward last-known exogenous readings — accuracy decreases "
                        "the further out the date is."
                    )

# ===== Tab 4: Actual vs Predicted (backtest) =====
with tab_backtest:
    st.subheader("Model Performance — Actual vs Predicted (Test Set)")

    predictions_df = load_predictions()

    if predictions_df is None:
        st.warning(
            "No backtest predictions found at `outputs/predictions.csv`. "
            "Run lstm_model.ipynb to generate them."
        )
    else:
        rmse = np.sqrt(mean_squared_error(predictions_df["Actual"], predictions_df["Predicted"]))
        mae = mean_absolute_error(predictions_df["Actual"], predictions_df["Predicted"])
        r2 = r2_score(predictions_df["Actual"], predictions_df["Predicted"])

        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE", f"{rmse:.4f}", help="Root Mean Squared Error — average error in kW, penalizes large misses more.")
        col2.metric("MAE", f"{mae:.4f}", help="Mean Absolute Error — average error in kW, in plain terms: 'off by this much on average.'")
        col3.metric("R² Score", f"{r2:.4f}", help="Proportion of variance explained by the model. Closer to 1.0 is better.")

        n_points = st.slider(
            "Number of test points to display", 100, min(5000, len(predictions_df)), 1000
        )
        window = predictions_df.head(n_points)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=window["Datetime"],
                y=window["Actual"],
                mode="lines",
                name="Actual",
                line=dict(color="#4C72B0"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=window["Datetime"],
                y=window["Predicted"],
                mode="lines",
                name="Predicted",
                line=dict(color="#DD8452"),
            )
        )
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="kW",
            height=450,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("View raw predictions"):
            st.dataframe(window, use_container_width=True)

st.markdown("---")
st.caption(
    "Model: 4-layer LSTM (hidden size 128) · Trained on UCI Household Power "
    "Consumption dataset · Sequence length: 60 minutes"
)
