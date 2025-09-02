# Fantasy Sports Analytics — Player Prop Edge Finder (Mock Project)

This is a **portfolio-ready** Python project that simulates a realistic workflow for identifying mispriced **player prop** lines across DFS/sportsbook platforms and **backtesting** a betting strategy.

> ⚠️ Data here is **synthetic** but designed to behave like real NBA player prop markets.

## What it does

- Loads historical (simulated) player prop lines + game outcomes
- Builds **predictive models** (baseline rolling averages + RandomForest)
- Flags **edges** where your prediction diverges from the posted line
- **Backtests** a simple Over/Under strategy using historical data
- Outputs ROI, win rate, and charts

## Folder Structure

```
fantasy_sports_analytics/
│── data/                 # simulated datasets (player props + results)
│── notebooks/            # (placeholder) exploration space
│── src/
│   ├── data_loader.py    # load & clean data
│   ├── model.py          # predictive models
│   ├── edge_finder.py    # compare predictions vs DFS lines
│   ├── backtest.py       # simulate betting strategy
│   ├── simulate_data.py  # generate synthetic historical data
│── results/
│   ├── figures/          # charts (ROI curve, edge distribution)
│   ├── reports/          # CSV summaries
│── README.md
```

## Quickstart

1. **Create a virtual env (optional)** and install dependencies (Python 3.9+ recommended):
   ```bash
   pip install -r requirements.txt
   ```
   Minimal libs used: `pandas`, `numpy`, `scikit-learn`, `matplotlib`.

2. **Generate data (already included in `data/` but you can re-generate):**
   ```bash
   python src/simulate_data.py
   ```

3. **Run backtest:**
   ```bash
   python src/backtest.py
   ```

4. **See results:** CSV reports + charts land in `results/`.

## Strategy (simplified)

- If model prediction is **above** line by a threshold → **bet Over**.
- If model prediction is **below** line by a threshold → **bet Under**.
- Otherwise **no bet**.
- Odds assumed at **-110** per side (risk 1 to win 0.9091).

You can tune thresholds, models, and payout assumptions in `src/backtest.py` and `src/model.py`.

## Notes

- This mock can be extended to real data by replacing `simulate_data.py` with scrapers/APIs and mapping fields to the same schema.
- The project aims to demonstrate **engineering + analytics** rigor (data pipeline, modeling, evaluation).

Enjoy!
