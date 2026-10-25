"""LangChain tool wrappers around the pure-Python functions in ``analysis.py``.

Key idea: the LLM never touches the DataFrame. It only sees each tool's
*name*, *docstring* and *typed arguments*, and decides which tool to call.
So the docstrings below are effectively prompts - write them clearly.

``make_tools(csv_path)`` returns the tools bound to one dataset (via closure),
so the model does not have to pass file paths or DataFrames around.
"""

from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.tools import BaseTool, tool

from .analysis import (
    compare_spending_periods as _compare,
    detect_unusual_expenses as _unusual,
    get_budget_recommendations as _budget,
    get_spending_summary_for_period as _summary,
    load_transactions,
)


def make_tools(csv_path: str | Path) -> List[BaseTool]:
    """Create the SpendWise tools bound to the transactions in ``csv_path``."""
    df = load_transactions(csv_path)  # validated once, reused by every tool

    @tool
    def get_spending_summary(
        start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> dict:
        """Summarize spending: total, number of transactions, totals per category
        and the largest transactions. Use for questions like "how much did I
        spend on X" or "what are my top categories".

        Args:
            start_date: Inclusive start date, format YYYY-MM-DD. Omit for all data.
            end_date: Inclusive end date, format YYYY-MM-DD. Omit for all data.
        """
        return _summary(df, start_date, end_date)

    @tool
    def compare_spending_periods(
        period_a_start: str,
        period_a_end: str,
        period_b_start: str,
        period_b_end: str,
    ) -> dict:
        """Compare spending between two periods. Period A is the baseline
        (e.g. last month), period B is the period being investigated (e.g. this
        month). Returns total change, % change, a per-category breakdown sorted
        by the biggest change, and supporting transactions. Use for "why did my
        spending increase/decrease?" questions. All dates are YYYY-MM-DD.
        """
        return _compare(df, period_a_start, period_a_end, period_b_start, period_b_end)

    @tool
    def detect_unusual_expenses() -> dict:
        """Find unusually large transactions compared with the typical amounts in
        the same category. Use for "any strange or unexpectedly big expenses?"."""
        return _unusual(df)

    @tool
    def get_budget_recommendations(
        budgets: Optional[Dict[str, float]] = None,
    ) -> dict:
        """Suggest monthly budgets per category from historical spending.

        Args:
            budgets: Optional monthly limits the user gave, e.g. {"Food": 400}.
                Do NOT invent limits; omit if the user did not provide any.
        This only suggests; it does not save anything.
        """
        return _budget(df, budgets)

    return [
        get_spending_summary,
        compare_spending_periods,
        detect_unusual_expenses,
        get_budget_recommendations,
    ]
