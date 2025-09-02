
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import random

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Seed for reproducibility
random.seed(42)
np.random.seed(42)

# --- Config ---
players = [
    "LeBron James", "Stephen Curry", "Kevin Durant", "Nikola Jokic",
    "Luka Doncic", "Jayson Tatum", "Damian Lillard", "Joel Embiid",
    "Kyrie Irving", "Anthony Davis", "Shai Gilgeous-Alexander",
    "Giannis Antetokounmpo", "Devin Booker", "Kawhi Leonard"
]
props = ["points", "rebounds", "assists"]
start_date = datetime(2022, 10, 18)
num_days = 180  # ~regular season length

# True skill parameters per player/prop (latent means + volatility)
latent = []
for p in players:
    for stat in props:
        base = {
            "points": np.random.uniform(18, 33),
            "rebounds": np.random.uniform(5, 12),
            "assists": np.random.uniform(4, 10),
        }[stat]
        vol = {"points": 6.0, "rebounds": 3.0, "assists": 2.5}[stat]
        latent.append({"player": p, "prop": stat, "base_mean": base, "game_sd": vol})

latent_df = pd.DataFrame(latent)

rows = []
for d in range(num_days):
    date = start_date + timedelta(days=int(d))
    # Each day, sample a subset of players who "play"
    todays_players = random.sample(players, k=random.randint(8, len(players)))
    for p in todays_players:
        for stat in props:
            row_latent = latent_df[(latent_df["player"] == p) & (latent_df["prop"] == stat)].iloc[0]
            # true performance for the day
            perf = np.random.normal(row_latent["base_mean"], row_latent["game_sd"])
            perf = max(0.0, perf)  # no negatives
            # Platform line is noisy estimate of latent mean (not the specific day's perf)
            market_noise = np.random.normal(0, row_latent["game_sd"] * 0.35)
            posted_line = max(0.0, row_latent["base_mean"] + market_noise)
            # Save
            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "player": p,
                "prop": stat,
                "platform": "MockBooks",
                "line": round(posted_line * 2) / 2.0,  # half-point lines
                "actual": perf
            })

df = pd.DataFrame(rows).sort_values(["date", "player", "prop"]).reset_index(drop=True)
df.to_csv(os.path.join(DATA_DIR, "player_props_history.csv"), index=False)

print(f"Saved {len(df)} rows to data/player_props_history.csv")
