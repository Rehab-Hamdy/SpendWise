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