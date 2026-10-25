"""Analysis utilities for SpendWise.

This module provides pure‑Python functions (no LLM) for loading a CSV of
transactions, validating the data, and computing basic spending summaries.

The functions are deliberately small and typed so they can be unit‑tested
independently of the LangChain agent that will wrap them later.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List

# Expected columns in the CSV file
EXPECTED_COLUMNS = ["date", "description", "category", "amount"]


def load_transactions(csv_path: str | Path) -> pd.DataFrame:
    """Load a CSV file containing transaction records.

    The CSV must contain the columns defined in ``EXPECTED_COLUMNS``. The
    ``date`` column is parsed as a ``datetime`` object and ``amount`` is cast to
    ``float``. Any missing values raise a ``ValueError``.

    Parameters
    ----------
    csv_path:
        Path to the CSV file.

    Returns
    -------
    pandas.DataFrame
        A validated DataFrame with appropriate dtypes.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Transaction file not found: {path}")

    # Load raw CSV
    df = pd.read_csv(path)

    # Validate column presence
    missing = set(EXPECTED_COLUMNS) - set(df.columns.str.lower())
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Normalise column names to lower case for internal consistency
    df.columns = [c.lower() for c in df.columns]

    # Parse dates
    try:
        df["date"] = pd.to_datetime(df["date"], errors="raise")
    except Exception as exc:
        raise ValueError(f"Invalid date values: {exc}") from exc

    # Ensure amount is numeric
    try:
        df["amount"] = pd.to_numeric(df["amount"], errors="raise")
    except Exception as exc:
        raise ValueError(f"Invalid amount values: {exc}") from exc

    # Check for any missing values in required columns
    if df[EXPECTED_COLUMNS].isnull().any().any():
        rows = df[EXPECTED_COLUMNS].isnull().any(axis=1)
        raise ValueError(
            f"Missing values detected in rows: {list(df[rows].index)}"
        )

    return df


def get_spending_summary(df: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
    """Calculate total spending and per‑category totals.

    Parameters
    ----------
    df:
        A DataFrame returned by :func:`load_transactions`.

    Returns
    -------
    total_spending:
        Sum of the ``amount`` column.
    category_totals:
        Mapping of category name → total amount spent in that category.
    """
    total_spending = df["amount"].sum()
    category_totals = df.groupby("category")["amount"].sum().to_dict()
    # Round for nicer display (optional)
    total_spending = round(total_spending, 2)
    category_totals = {k: round(v, 2) for k, v in category_totals.items()}
    return total_spending, category_totals


def validate_and_summarise(csv_path: str | Path) -> Tuple[float, Dict[str, float]]:
    """Convenient helper used by the initial demo script.

    Loads the CSV, validates it, and returns the spending summary.
    """
    df = load_transactions(csv_path)
    return get_spending_summary(df)

def get_spending_summary_for_period(
    df: pd.DataFrame, start: str | None = None, end: str | None = None
) -> dict:
    """Phase 2 version of the summary: optional date range, richer output.

    Parameters
    ----------
    df: validated transactions DataFrame.
    start, end: inclusive ISO dates (``"YYYY-MM-DD"``). ``None`` means unbounded.

    Returns a JSON-friendly dict (total, transaction count, per-category totals
    sorted from largest to smallest, and the 5 largest transactions).
    """
    data = _filter_period(df, start, end)
    cats = data.groupby("category")["amount"].sum().sort_values(ascending=False)
    top = data.nlargest(5, "amount")
    return {
        "start": start,
        "end": end,
        "total": round(float(data["amount"].sum()), 2),
        "transaction_count": int(len(data)),
        "category_totals": {k: round(float(v), 2) for k, v in cats.items()},
        "top_transactions": _records(top),
    }


def compare_spending_periods(
    df: pd.DataFrame,
    period_a_start: str,
    period_a_end: str,
    period_b_start: str,
    period_b_end: str,
) -> dict:
    """Compare spending in period B (the "new" one) against period A (the "base").

    Returns totals, absolute/percent change, and a per-category breakdown sorted
    by the size of the change, so "what drove the increase?" is answered by the
    first entry. Also includes the biggest transactions in the top-changing
    category of period B as supporting evidence.
    """
    a = _filter_period(df, period_a_start, period_a_end)
    b = _filter_period(df, period_b_start, period_b_end)

    cat_a = a.groupby("category")["amount"].sum()
    cat_b = b.groupby("category")["amount"].sum()
    categories = sorted(set(cat_a.index) | set(cat_b.index))

    breakdown = []
    for cat in categories:
        va, vb = float(cat_a.get(cat, 0.0)), float(cat_b.get(cat, 0.0))
        breakdown.append(
            {
                "category": cat,
                "period_a": round(va, 2),
                "period_b": round(vb, 2),
                "change": round(vb - va, 2),
                "pct_change": _pct(va, vb),
            }
        )
    breakdown.sort(key=lambda r: abs(r["change"]), reverse=True)

    total_a, total_b = float(a["amount"].sum()), float(b["amount"].sum())
    supporting = []
    if breakdown and breakdown[0]["change"] != 0:
        top_cat = breakdown[0]["category"]
        supporting = _records(b[b["category"] == top_cat].nlargest(3, "amount"))

    return {
        "period_a": {"start": period_a_start, "end": period_a_end, "total": round(total_a, 2)},
        "period_b": {"start": period_b_start, "end": period_b_end, "total": round(total_b, 2)},
        "total_change": round(total_b - total_a, 2),
        "total_pct_change": _pct(total_a, total_b),
        "category_breakdown": breakdown,
        "supporting_transactions": supporting,
    }


def detect_unusual_expenses(
    df: pd.DataFrame, iqr_multiplier: float = 1.5, min_transactions: int = 4
) -> dict:
    """Flag unusually large transactions using the IQR rule, per category.

    A transaction is unusual if ``amount > Q3 + iqr_multiplier * IQR`` of its
    category. Categories with fewer than ``min_transactions`` rows are skipped
    (too little history to call anything "unusual").
    """
    flagged = []
    skipped = []
    for cat, group in df.groupby("category"):
        if len(group) < min_transactions:
            skipped.append(cat)
            continue
        q1, q3 = group["amount"].quantile([0.25, 0.75])
        limit = q3 + iqr_multiplier * (q3 - q1)
        typical = float(group["amount"].median())
        for _, row in group[group["amount"] > limit].iterrows():
            flagged.append(
                {
                    "date": row["date"].strftime("%Y-%m-%d"),
                    "description": row["description"],
                    "category": cat,
                    "amount": round(float(row["amount"]), 2),
                    "category_median": round(typical, 2),
                    "threshold": round(float(limit), 2),
                }
            )
    flagged.sort(key=lambda r: r["amount"], reverse=True)
    return {"unusual_transactions": flagged, "skipped_categories": skipped}


def get_budget_recommendations(
    df: pd.DataFrame, budgets: Dict[str, float] | None = None, reduction_pct: float = 10.0
) -> dict:
    """Suggest monthly budgets from historical spending.

    For each category the average monthly spend is computed. If the user gave a
    ``budgets`` limit (monthly) for it, we report over/under and how much to cut.
    Without a limit, we suggest ``reduction_pct`` % below the average.
    This function only *suggests*; it never saves anything (saving comes in
    Phase 5 with human confirmation).
    """
    budgets = budgets or {}
    monthly = df.groupby([df["date"].dt.to_period("M"), "category"])["amount"].sum()
    n_months = df["date"].dt.to_period("M").nunique()
    avg = (monthly.groupby("category").sum() / n_months).sort_values(ascending=False)

    recs = []
    for cat, avg_spend in avg.items():
        avg_spend = float(avg_spend)
        limit = budgets.get(cat)
        if limit is not None:
            diff = avg_spend - limit
            status = "over_budget" if diff > 0 else "within_budget"
            suggested = round(limit, 2)
            action = (
                f"Reduce by about {diff:.2f} per month to meet the {limit:.2f} limit."
                if diff > 0
                else f"On track; {abs(diff):.2f} below the limit on average."
            )
        else:
            status = "no_budget_set"
            suggested = round(avg_spend * (1 - reduction_pct / 100), 2)
            action = f"Consider a budget of {suggested:.2f} ({reduction_pct:.0f}% below average)."
        recs.append(
            {
                "category": cat,
                "avg_monthly_spend": round(avg_spend, 2),
                "budget": limit,
                "status": status,
                "suggested_monthly_budget": suggested,
                "recommendation": action,
            }
        )
    return {"months_analyzed": int(n_months), "recommendations": recs}


# ---------------------------------------------------------------- helpers ---
def _filter_period(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    """Return rows with start <= date <= end (inclusive)."""
    out = df
    if start:
        out = out[out["date"] >= pd.to_datetime(start)]
    if end:
        out = out[out["date"] <= pd.to_datetime(end)]
    return out


def _pct(old: float, new: float) -> float | None:
    """Percent change; None when the base is zero (undefined)."""
    if old == 0:
        return None
    return round((new - old) / old * 100, 2)


def _records(data: pd.DataFrame) -> List[dict]:
    """Convert transaction rows to JSON-friendly dicts."""
    return [
        {
            "date": r["date"].strftime("%Y-%m-%d"),
            "description": r["description"],
            "category": r["category"],
            "amount": round(float(r["amount"]), 2),
        }
        for _, r in data.iterrows()
    ]


# Example usage when run as a script (not imported)
if __name__ == "__main__":
    import argparse, json, sys

    parser = argparse.ArgumentParser(description="Load transactions CSV and print a summary.")
    parser.add_argument("csv_path", help="Path to the transactions CSV file")
    args = parser.parse_args()

    try:
        total, cat_totals = validate_and_summarise(args.csv_path)
        summary = {"total_spending": total, "category_totals": cat_totals}
        json.dump(summary, sys.stdout, indent=2)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)