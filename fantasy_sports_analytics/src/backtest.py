
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data_loader import load_history, add_features
from model import baseline_predict, fit_ml_model, ml_predict
from edge_finder import compute_edges

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
REPORT_DIR = os.path.join(RESULTS_DIR, "reports")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

def simulate_bets(df, signal_col="signal", pred_col="pred", line_col="line", actual_col="actual", odds=-110):
    """Simulate flat 1-unit bets; American odds assumed per side."""
    df = df.copy()
    # Convert American odds to decimal payout per unit
    if odds < 0:
        payout = 100 / abs(odds)  # e.g., -110 -> 0.9091 win per 1 risked
    else:
        payout = odds / 100.0

    def result(row):
        if row[signal_col] == "OVER":
            return 1 if row[actual_col] > row[line_col] else 0
        if row[signal_col] == "UNDER":
            return 1 if row[actual_col] < row[line_col] else 0
        return None

    df["win"] = df.apply(result, axis=1)
    df = df[~df["win"].isna()].copy()
    df["stake"] = 1.0
    df["pl"] = np.where(df["win"] == 1, payout, -1.0)

    # Equity curve
    df = df.sort_values("date")
    df["cum_pl"] = df["pl"].cumsum()
    return df, payout

def main():
    # Load + feature engineering
    hist = load_history(DATA_DIR)
    hist = add_features(hist)

    # Train/validation split by date (time-aware split)
    cutoff_date = hist["date"].quantile(0.6)
    train = hist[hist["date"] <= cutoff_date].copy()
    test = hist[hist["date"] > cutoff_date].copy()

    # Baseline predictions
    train["baseline_pred"] = baseline_predict(train)
    test["baseline_pred"] = baseline_predict(test)

    # ML model (trained on baseline features)
    model = fit_ml_model(train)
    train["ml_pred"] = train["baseline_pred"] * 0  # unused; for symmetry
    test["ml_pred"] = ml_predict(model, test)

    # Choose which prediction to use for edges (you can switch to 'ml_pred')
    use_pred_col = "baseline_pred"

    # Compute edges & signals
    threshold = 1.0  # half a bucket of value required
    test = compute_edges(test, pred_col=use_pred_col, threshold=threshold)

    # Backtest
    bt, payout = simulate_bets(test, signal_col="signal", pred_col=use_pred_col)

    # Summary
    total_bets = len(bt)
    wins = int(bt["win"].sum())
    losses = total_bets - wins
    roi = bt["pl"].sum() / total_bets if total_bets > 0 else 0.0
    hit_rate = wins / total_bets if total_bets > 0 else 0.0

    summary = {
        "total_bets": total_bets,
        "wins": wins,
        "losses": losses,
        "hit_rate": round(hit_rate * 100, 2),
        "roi_per_bet": round(roi * 100, 2),
        "assumed_odds": -110,
        "edge_threshold": threshold,
        "prediction_source": use_pred_col
    }
    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(REPORT_DIR, "backtest_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    # Save detailed bets
    detail_cols = ["date", "player", "prop", "line", "actual", use_pred_col, "edge", "signal", "win", "pl", "cum_pl"]
    details_path = os.path.join(REPORT_DIR, "backtest_details.csv")
    bt.merge(test[["date","player","prop","line","actual",use_pred_col,"edge","signal"]],
             on=["date","player","prop","line","actual"], how="left")[detail_cols]\
      .to_csv(details_path, index=False)

    # Charts (each chart its own plot; default colors only)
    # ROI curve
    plt.figure()
    bt.groupby("date")["pl"].sum().cumsum().plot()
    plt.title("Cumulative P&L over Time")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Units")
    plt.tight_layout()
    roi_path = os.path.join(FIG_DIR, "roi_curve.png")
    plt.savefig(roi_path, dpi=150)
    plt.close()

    # Edge distribution (only bets placed)
    plt.figure()
    test.loc[test["signal"].notna(), "edge"].hist(bins=30)
    plt.title("Edge Distribution (Placed Bets)")
    plt.xlabel("Model Prediction − Line")
    plt.ylabel("Count")
    plt.tight_layout()
    edge_path = os.path.join(FIG_DIR, "edge_distribution.png")
    plt.savefig(edge_path, dpi=150)
    plt.close()

    print("Backtest summary saved to:", summary_path)
    print("Backtest details saved to:", details_path)
    print("Figures saved to:", FIG_DIR)
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
