import io
import json
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ART_MODEL = Path("rf_power_model.joblib")
ART_FEATS = Path("feature_cols.json")
ART_STATE = Path("last_state.csv")
ART_LAST_TIME = Path("last_time.txt")

st.set_page_config(page_title="Power Consumption Forecast", layout="wide")

# ---------------------- helpers ----------------------
def read_uci_file(file_or_path) -> pd.DataFrame:
    """Load UCI household dataset and return raw dataframe with Datetime index."""
    df = pd.read_csv(file_or_path, sep=";", low_memory=False)
    df.replace("?", pd.NA, inplace=True)
    df.dropna(inplace=True)
    df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    df.set_index("Datetime", inplace=True)
    df.drop(columns=["Date", "Time"], inplace=True)
    df = df.apply(pd.to_numeric)
    return df

def make_hourly_target(df: pd.DataFrame) -> pd.Series:
    s = df["Global_active_power"].resample("h").mean()
    return s.interpolate(method="time")

def build_feature_frame(power_hourly: pd.Series) -> pd.DataFrame:
    df = power_hourly.to_frame(name="target")
    # calendar
    df["hour"] = df.index.hour
    df["day"] = df.index.day
    df["month"] = df.index.month
    df["weekday"] = df.index.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    # lags
    for i in range(1, 25):
        df[f"lag_{i}"] = df["target"].shift(i)
    # rolling
    df["rolling_3h"] = df["target"].rolling(3).mean()
    df["rolling_6h"] = df["target"].rolling(6).mean()
    df["rolling_12h"] = df["target"].rolling(12).mean()
    df["rolling_24h"] = df["target"].rolling(24).mean()
    return df.dropna()

def train_rf(df_features: pd.DataFrame) -> tuple[RandomForestRegressor, list]:
    X = df_features.drop(columns=["target"])
    y = df_features["target"]
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=20, random_state=42, n_jobs=-1
    ).fit(X, y)
    return rf, X.columns.tolist()

def save_artifacts(model, feature_cols, last_row: pd.Series, last_time: pd.Timestamp):
    dump(model, ART_MODEL)
    ART_FEATS.write_text(json.dumps(feature_cols))
    last_row.to_csv(ART_STATE)
    ART_LAST_TIME.write_text(str(last_time))

def load_artifacts():
    model = load(ART_MODEL)
    feature_cols = json.loads(ART_FEATS.read_text())

    last = pd.read_csv(ART_STATE, index_col=0, header=None).squeeze("columns")
    if isinstance(last, pd.DataFrame):
        last = last.iloc[:, 0]
    last.index = last.index.astype(str)

    # fix dtypes
    lag_cols  = [c for c in last.index if c.startswith("lag_")]
    roll_cols = [c for c in last.index if c.startswith("rolling_")]
    num_cols  = ["target"] + lag_cols + roll_cols
    last.loc[num_cols] = pd.to_numeric(last.loc[num_cols], errors="coerce")
    for c in ["hour", "day", "month", "weekday", "is_weekend"]:
        if c in last.index:
            last.loc[c] = int(float(last.loc[c]))

    last_time = pd.to_datetime(ART_LAST_TIME.read_text().strip())
    return model, feature_cols, last, last_time

def forecast_future_df(model, feature_cols, last_state: pd.Series, last_time: pd.Timestamp, steps=24) -> pd.Series:
    times, preds = [], []
    state = last_state.copy()
    for _ in range(steps):
        X_row = state[feature_cols].to_frame().T
        yhat = model.predict(X_row)[0]
        last_time = last_time + pd.Timedelta(hours=1)
        times.append(last_time); preds.append(yhat)

        # shift lags
        for k in range(24, 1, -1):
            state[f"lag_{k}"] = state[f"lag_{k-1}"]
        state["lag_1"] = yhat
        # rolling
        state["rolling_3h"]  = np.mean([state[f"lag_{j}"] for j in range(1, 4)])
        state["rolling_6h"]  = np.mean([state[f"lag_{j}"] for j in range(1, 7)])
        state["rolling_12h"] = np.mean([state[f"lag_{j}"] for j in range(1,13)])
        state["rolling_24h"] = np.mean([state[f"lag_{j}"] for j in range(1,25)])
        # calendar from new time
        state["hour"] = last_time.hour
        state["day"] = last_time.day
        state["month"] = last_time.month
        state["weekday"] = last_time.weekday()
        state["is_weekend"] = int(state["weekday"] >= 5)
    return pd.Series(preds, index=pd.DatetimeIndex(times, name="Datetime"), name="forecast_kW")

# ---------------------- UI ----------------------
st.title("⚡ City Power Consumption Forecast (Hourly)")

with st.sidebar:
    st.header("Controls")
    mode = st.radio("Mode", ["Use saved model", "Train from dataset"], index=0)
    horizon_days = st.slider("Forecast horizon (days)", 1, 14, 7)
    horizon = horizon_days * 24

    st.markdown("---")
    st.caption("Dataset (UCI household power usage). Provide the raw text file or a preprocessed CSV.")
    uploaded = st.file_uploader("Upload dataset file (optional)", type=["txt", "csv"])

# ---------------------- main flow ----------------------
if mode == "Train from dataset":
    if uploaded is None:
        st.info("Upload the UCI file (e.g., `household_power_consumption.txt`) to train.")
        st.stop()

    # load & prep
    with st.spinner("Loading & cleaning data..."):
        raw = read_uci_file(uploaded)
        hourly = make_hourly_target(raw)
        feats = build_feature_frame(hourly)

    # split for quick report
    split = int(len(feats)*0.8)
    train, test = feats.iloc[:split], feats.iloc[split:]
    Xtr, ytr = train.drop(columns=["target"]), train["target"]
    Xte, yte = test.drop(columns=["target"]),  test["target"]

    with st.spinner("Training RandomForest..."):
        model, cols = train_rf(train)

    # eval
    pred = model.predict(Xte)
    mae = mean_absolute_error(yte, pred)
    rmse = (mean_squared_error(yte, pred))**0.5

    st.success(f"Model trained ✅  |  MAE: **{mae:.3f}**  RMSE: **{rmse:.3f}**")

    # save artifacts built on ALL data
    with st.spinner("Fitting on all data & saving artifacts..."):
        model_all, cols_all = train_rf(feats)
        save_artifacts(model_all, cols_all, feats.iloc[-1], feats.index[-1])
    st.caption("Artifacts saved: rf_power_model.joblib, feature_cols.json, last_state.csv, last_time.txt")

    # forecast & plot
    fc = forecast_future_df(model_all, cols_all, feats.iloc[-1], feats.index[-1], steps=horizon)
    st.subheader("History vs Forecast")
    hist = hourly.iloc[-7*24:]  # last week actuals
    combo = pd.concat([hist.rename("actual_kW"), fc], axis=1)
    st.line_chart(combo)

    csv = fc.to_csv().encode()
    st.download_button("Download forecast CSV", csv, file_name="forecast.csv", mime="text/csv")

else:  # Use saved model
    if not (ART_MODEL.exists() and ART_FEATS.exists() and ART_STATE.exists() and ART_LAST_TIME.exists()):
        st.warning("No saved artifacts found. Switch to **Train from dataset** first.")
        st.stop()

    model, cols, last_state, last_time = load_artifacts()
    st.success("Loaded saved model and state ✅")

    fc = forecast_future_df(model, cols, last_state, last_time, steps=horizon)
    st.subheader("Forecast")
    st.line_chart(fc)

    # Optional: overlay last week actuals if user gives the dataset
    if uploaded is not None:
        raw = read_uci_file(uploaded)
        hourly = make_hourly_target(raw)
        hist = hourly.iloc[-7*24:]
        combo = pd.concat([hist.rename("actual_kW"), fc], axis=1)
        st.subheader("History (last 7d) + Forecast")
        st.line_chart(combo)

    csv = fc.to_csv().encode()
    st.download_button("Download forecast CSV", csv, file_name="forecast.csv", mime="text/csv")

st.markdown("---")
st.caption("Dataset source: UCI Individual Household Electric Power Consumption.")
