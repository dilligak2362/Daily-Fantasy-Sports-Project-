
from typing import Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

BASELINE_FEATURES = ["roll_mean_3", "roll_mean_5", "roll_mean_10", "season_mean", "lag1"]

def baseline_predict(df: pd.DataFrame) -> pd.Series:
    """A simple ensemble baseline using rolling means + lag."""
    # Weighted average: more weight on recent windows
    preds = (
        0.45 * df["roll_mean_3"] +
        0.30 * df["roll_mean_5"] +
        0.15 * df["roll_mean_10"] +
        0.05 * df["season_mean"] +
        0.05 * df["lag1"].fillna(df["roll_mean_3"])
    )
    return preds

def fit_ml_model(train_df: pd.DataFrame) -> RandomForestRegressor:
    X = train_df[BASELINE_FEATURES]
    y = train_df["actual"]
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    return model

def ml_predict(model: RandomForestRegressor, df: pd.DataFrame) -> pd.Series:
    X = df[BASELINE_FEATURES]
    return pd.Series(model.predict(X), index=df.index)
