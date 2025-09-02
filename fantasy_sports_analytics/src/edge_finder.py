
import pandas as pd

def compute_edges(df: pd.DataFrame, pred_col: str, threshold: float = 1.0) -> pd.DataFrame:
    """Compute edge and generate signals: 'OVER', 'UNDER', or None based on threshold difference."""
    out = df.copy()
    out["edge"] = out[pred_col] - out["line"]
    def signal(row):
        if row["edge"] >= threshold:
            return "OVER"
        elif row["edge"] <= -threshold:
            return "UNDER"
        return None
    out["signal"] = out.apply(signal, axis=1)
    return out
