import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "notebook", "household_power_consumption.txt")
MODEL = os.path.join(BASE, "models", "best_model.pth")
SCALER = os.path.join(BASE, "models", "scaler.pkl")
OUT = os.path.join(BASE, "outputs", "predictions.csv")

SEQ = 60
TARGET = "Global_active_power"
FEATURES = ["Global_reactive_power", "Voltage", "Global_intensity",
            "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
            "Hour", "Day", "Month", "Year", "Weekday"]
TEST_LIMIT = None   # e.g. 50000 for a quick run on the last 50k test points


class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=4, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


# --- load + preprocess (same as app.py) ---
df = pd.read_csv(DATA, sep=";", na_values="?", low_memory=False)
for c in [TARGET, "Global_reactive_power", "Voltage", "Global_intensity",
          "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
df = df.drop(columns=["Date", "Time"]).set_index("Datetime").sort_index()
df = df.ffill().dropna()
df["Hour"], df["Day"], df["Month"] = df.index.hour, df.index.day, df.index.month
df["Year"], df["Weekday"] = df.index.year, df.index.dayofweek

scaler = joblib.load(SCALER)
X = scaler.transform(df[FEATURES].values).astype(np.float32)
y = df[TARGET].values.astype(np.float32)
times = df.index

# --- sequences: window X[i:i+60] predicts y[i+60]; time-ordered 80/20 split ---
n_seq = len(df) - SEQ
split = int(0.8 * n_seq)
start = max(split, n_seq - TEST_LIMIT) if TEST_LIMIT else split
windows = np.lib.stride_tricks.sliding_window_view(X, SEQ, axis=0)  # (N-59, 11, 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMForecaster(len(FEATURES))
model.load_state_dict(torch.load(MODEL, map_location=device))
model.to(device).eval()

preds = []
BATCH = 2048
with torch.no_grad():
    for b in range(start, n_seq, BATCH):
        e = min(b + BATCH, n_seq)
        batch = torch.from_numpy(np.ascontiguousarray(windows[b:e].transpose(0, 2, 1)))
        preds.append(model(batch.to(device)).cpu().numpy())
        if (b - start) // BATCH % 20 == 0:
            print(f"{e - start:,} / {n_seq - start:,}")

preds = np.concatenate(preds)
idx = np.arange(start, n_seq) + SEQ
os.makedirs(os.path.dirname(OUT), exist_ok=True)
pd.DataFrame({"Datetime": times[idx], "Actual": y[idx], "Predicted": preds}).to_csv(OUT, index=False)
print("Saved", OUT)