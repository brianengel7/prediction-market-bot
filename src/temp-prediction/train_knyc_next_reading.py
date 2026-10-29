
#!/usr/bin/env python3
"""Offline KNYC next-hourly-reading forecast comparison.

Reads existing CSV files only. Does not use Kalshi APIs,
place orders, access GitHub, or modify existing data.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


BASE = Path(
    "data/research/nyc_observation_20260701_20261026"
)
OUT = BASE / "model_research"

# ----------------------------------------------------------
# LOAD ONE-MINUTE KNYC MEASUREMENTS
# ----------------------------------------------------------

minute = pd.read_csv(
    BASE / "knyc_asos1min_20260701_20261026.csv"
)

minute["valid_utc"] = pd.to_datetime(
    minute["valid_utc"], utc=True, errors="coerce"
)
minute["temperature_f"] = pd.to_numeric(
    minute["temperature_f"], errors="coerce"
)

minute = (
    minute.dropna(subset=["valid_utc", "temperature_f"])
    .sort_values("valid_utc")
)

if minute["valid_utc"].duplicated().any():
    raise ValueError("Duplicate one-minute timestamps")

mt = pd.DatetimeIndex(minute["valid_utc"]).as_unit("ns").asi8
mv = minute["temperature_f"].to_numpy(dtype=float)

# ----------------------------------------------------------
# LOAD OFFICIAL HOURLY OBSERVATIONS (OUR TARGETS)
# ----------------------------------------------------------

hourly = pd.read_csv(
    BASE / "knyc_high_frequency_observations.csv"
)

if not {"valid", "tmpf"}.issubset(hourly.columns):
    raise ValueError(
        f"Unexpected hourly columns: {hourly.columns.tolist()}"
    )

hourly["observed_utc"] = pd.to_datetime(
    hourly["valid"], utc=True, errors="coerce"
)
hourly["target_f"] = pd.to_numeric(
    hourly["tmpf"], errors="coerce"
)

hourly = hourly.dropna(
    subset=["observed_utc", "target_f"]
)

# Restrict to the scheduled routine observation minute.
hourly = hourly[
    hourly["observed_utc"].dt.minute == 51
].copy()

if "station" in hourly.columns:
    hourly = hourly[
        hourly["station"].astype(str).str.upper()
        .isin(["NYC", "KNYC"])
    ]

# Exclude conflicting versions of the same observation.
conflict = hourly.groupby(
    "observed_utc"
)["target_f"].nunique()

hourly = hourly[
    ~hourly["observed_utc"].isin(
        conflict[conflict > 1].index
    )
]

hourly = (
    hourly.sort_values("observed_utc")
    .drop_duplicates("observed_utc")
)

ht = pd.DatetimeIndex(hourly["observed_utc"]).as_unit("ns").asi8
hv = hourly["target_f"].to_numpy(dtype=float)

MIN_NS = 60 * 1_000_000_000


def at_or_before(end_ns, max_age_minutes=3):
    """Return last measurement at or before cutoff.

    Never use a measurement timestamped after the cutoff.
    """
    j = int(
        np.searchsorted(mt, end_ns, side="right") - 1
    )

    if j < 0 or end_ns - mt[j] > max_age_minutes * MIN_NS:
        return np.nan

    return float(mv[j])


# ----------------------------------------------------------
# BUILD ONE PREDICTION SAMPLE PER HOURLY OBSERVATION
# ----------------------------------------------------------

rows = []

for r in hourly.itertuples(index=False):
    ts = pd.Timestamp(r.observed_utc)

    # Simulated decision: 5 min before measured observation.
    decision_ns = int(ts.value - 5 * MIN_NS)

    now = at_or_before(decision_ns)

    if not np.isfinite(now):
        continue

    # Require previous official report to be sufficiently old.
    prior_i = int(
        np.searchsorted(
            ht,
            decision_ns - 10 * MIN_NS,
            side="right",
        ) - 1
    )

    if prior_i < 0:
        continue

    prior_age_minutes = (
        decision_ns - ht[prior_i]
    ) / MIN_NS

    if not 20 <= prior_age_minutes <= 180:
        continue

    prev = float(hv[prior_i])

    recent = {}

    for offset in [5, 15, 30, 60]:
        old = at_or_before(
            decision_ns - offset * MIN_NS
        )

        recent[f"delta_{offset}m"] = (
            now - old if np.isfinite(old) else np.nan
        )

    win_start = np.searchsorted(
        mt, decision_ns - 15 * MIN_NS, side="left"
    )
    win_end = np.searchsorted(
        mt, decision_ns, side="right"
    )

    win = mv[win_start:win_end]
    enough = len(win) >= 5

    local = ts.tz_convert("America/New_York")

    rows.append({
        "observed_at_utc": ts.isoformat(),
        "cutoff_at_utc": pd.Timestamp(
            decision_ns, tz="UTC"
        ).isoformat(),
        "local_date": str(local.date()),
        "target_f": float(r.target_f),
        "latest_minute_f": now,
        "previous_report_f": prev,
        "minute_minus_previous_report": now - prev,
        **recent,
        "mean_15m": (
            float(np.mean(win)) if enough else np.nan
        ),
        "std_15m": (
            float(np.std(win)) if enough else np.nan
        ),
        "min_15m": (
            float(np.min(win)) if enough else np.nan
        ),
        "max_15m": (
            float(np.max(win)) if enough else np.nan
        ),
        "hour_sin": float(
            np.sin(2 * np.pi * local.hour / 24)
        ),
        "hour_cos": float(
            np.cos(2 * np.pi * local.hour / 24)
        ),
        "year_sin": float(
            np.sin(2 * np.pi * local.dayofyear / 365.25)
        ),
        "year_cos": float(
            np.cos(2 * np.pi * local.dayofyear / 365.25)
        ),
    })

data = pd.DataFrame(rows)

if data.empty:
    raise RuntimeError(
        "No matched observations. Check source timestamps."
    )

features = [
    x for x in data.columns
    if x not in {
        "observed_at_utc",
        "cutoff_at_utc",
        "local_date",
        "target_f",
    }
]

# ----------------------------------------------------------
# CHRONOLOGICAL TRAIN / VALIDATION / TEST
# ----------------------------------------------------------

train = data[
    data["local_date"] <= "2026-08-31"
].copy()

valid = data[
    (data["local_date"] >= "2026-09-01")
    & (data["local_date"] <= "2026-09-30")
].copy()

test = data[
    (data["local_date"] >= "2026-10-01")
    & (data["local_date"] <= "2026-10-26")
].copy()

if min(len(train), len(valid), len(test)) < 30:
    raise RuntimeError(
        "Insufficient samples in one split: "
        f"{len(train)}, {len(valid)}, {len(test)}"
    )

# Predict temperature ADJUSTMENT relative to latest minute.
models = {}

for leaves in [7, 15, 31]:
    m = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=leaves,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42,
    )

    m.fit(
        train[features],
        train["target_f"] - train["latest_minute_f"],
    )

    pred = (
        valid["latest_minute_f"].to_numpy()
        + m.predict(valid[features])
    )

    models[leaves] = (
        mean_absolute_error(valid["target_f"], pred),
        m,
    )

chosen, (validation_mae, model) = min(
    models.items(),
    key=lambda item: item[1][0],
)


def evaluate(label, actual, prediction):
    errors = np.abs(actual - prediction)

    return {
        "model": label,
        "n": len(errors),
        "MAE_F": round(float(errors.mean()), 3),
        "within_0.5F": round(
            float((errors <= 0.5).mean()), 3
        ),
        "within_1F": round(
            float((errors <= 1).mean()), 3
        ),
        "within_2F": round(
            float((errors <= 2).mean()), 3
        ),
    }


# ----------------------------------------------------------
# UNSEEN OCTOBER PERFORMANCE
# ----------------------------------------------------------

actual = test["target_f"].to_numpy()
now = test["latest_minute_f"].to_numpy()

trend = test["delta_15m"].fillna(0).to_numpy()

pred_model = now + model.predict(test[features])

results = [
    evaluate(
        "previous_hourly_report",
        actual,
        test["previous_report_f"].to_numpy(),
    ),
    evaluate(
        "latest_archived_minute",
        actual,
        now,
    ),
    evaluate(
        "minute_linear_trend",
        actual,
        now + (5 / 15) * trend,
    ),
    evaluate(
        "boosted_model",
        actual,
        pred_model,
    ),
]

# ----------------------------------------------------------
# SAVE RESEARCH RESULTS
# ----------------------------------------------------------

OUT.mkdir(parents=True, exist_ok=True)

test["prediction_f"] = pred_model
test["prediction_error_f"] = pred_model - actual

test.to_csv(
    OUT / "october_predictions.csv",
    index=False,
)

pd.DataFrame(results).to_csv(
    OUT / "model_comparison.csv",
    index=False,
)

print("\n=== KNYC NEXT READING FORECAST ===")
print("Historical one-minute rows:", len(minute))
print("Hourly :51 targets:", len(hourly))
print("Matched target events:", len(data))
print(
    "Train (Jul-Aug):", len(train),
    "Validation (Sep):", len(valid),
    "Test (Oct):", len(test),
)
print("Best leaf count selected on September:", chosen)
print("September validation MAE:", round(validation_mae, 3))

print("\n=== UNSEEN OCTOBER PERFORMANCE ===")
print(pd.DataFrame(results).to_string(index=False))

print("\nSaved:", OUT / "october_predictions.csv")
print("Saved:", OUT / "model_comparison.csv")

print(
    "\nWARNING: Archived 1-minute temperatures were "
    "not necessarily available live at cutoff."
)
print(
    "This tests potential predictive skill, "
    "NOT an executable trading strategy."
)
