
import os
import pandas as pd

def load_history(data_dir: str):
    """Load historical simulated props dataset."""
    path = os.path.join(data_dir, "player_props_history.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    return df

def add_features(df: pd.DataFrame):
    """Add rolling features per player/prop using past k games."""
    df = df.sort_values(["player", "prop", "date"]).copy()
    # Rolling means and stds using only past data (shifted)
    for k in [3, 5, 10]:
        df[f"roll_mean_{k}"] = (
            df.groupby(["player", "prop"])["actual"]
              .rolling(k, min_periods=1).mean().shift(1).reset_index(level=[0,1], drop=True)
        )
        df[f"roll_std_{k}"] = (
            df.groupby(["player", "prop"])["actual"]
              .rolling(k, min_periods=1).std().shift(1).reset_index(level=[0,1], drop=True)
        )
    # Season avg to date
    df["season_mean"] = (
        df.groupby(["player", "prop"])["actual"]
          .expanding().mean().shift(1).reset_index(level=[0,1], drop=True)
    )
    # Use previous game's actual as a recency signal
    df["lag1"] = (
        df.groupby(["player", "prop"])["actual"]
          .shift(1)
    )
    # Fill NaNs in features gracefully
    feature_cols = [c for c in df.columns if c.startswith(("roll_mean_", "roll_std_", "season_mean", "lag1"))]
    for c in feature_cols:
        df[c] = df[c].fillna(method="bfill").fillna(df[c].median())
    return df
