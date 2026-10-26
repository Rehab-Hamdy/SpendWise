"""LangChain tool wrappers around the pure-Python functions in ``analysis.py``.

Key idea: the LLM never touches the DataFrame. It only sees each tool's
*name*, *docstring* and *typed arguments*, and decides which tool to call.
So the docstrings below are effectively prompts - write them clearly.

``make_tools(csv_path, budgets_path)`` returns the tools bound to one dataset
and budget file (via closure), so the model does not have to pass file paths or
DataFrames around.
"""

from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command

from .analysis import (
    DEFAULT_BUDGETS_PATH,
    compare_spending_periods as _compare,
    delete_budget_limit as _delete_budget,
    detect_unusual_expenses as _unusual,
    get_budget_recommendations as _budget,
    get_dataset_overview as _overview,
    get_spending_summary_for_period as _summary,
    load_budgets,
    load_transactions,
    save_budget_limit as _save_budget,
)


def make_tools(
    csv_path: str | Path,
    budgets_path: str | Path = DEFAULT_BUDGETS_PATH,
) -> List[BaseTool]:
    """Create the SpendWise tools bound to ``csv_path`` and ``budgets_path``."""
    df = load_transactions(csv_path)  # validated once, reused by every tool
    known_categories = sorted(df["category"].unique())

    @tool
    def get_dataset_overview() -> dict:
        """Return dataset overview: exact earliest_date, latest_date, total
        transaction_count, total_spending, and available categories. Use when
        asked about what dates or categories the dataset covers."""
        return _overview(df)

    @tool
    def get_spending_summary(
        start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> dict:
        """Summarize spending: total, number of transactions, earliest_date and
        latest_date of matched transactions, totals per category, and the
        largest transactions. Use for questions like "how much did I spend on X"
        or "what are my top categories".

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
        runtime: ToolRuntime,
    ) -> Command:
        """Compare spending between two periods. Period A is the baseline
        (e.g. last month), period B is the period being investigated (e.g. this
        month). Returns total change, % change, a per-category breakdown sorted
        by the biggest change, and supporting transactions. Use for "why did my
        spending increase/decrease?" questions. All dates are YYYY-MM-DD.
        """
        result = _compare(df, period_a_start, period_a_end, period_b_start, period_b_end)
        # A tool can update the agent's *state* by returning a Command.
        # `runtime` is injected by LangChain and is hidden from the model.
        return Command(
            update={
                "last_comparison": result,  # remembered for follow-up questions
                "messages": [
                    ToolMessage(content=str(result), tool_call_id=runtime.tool_call_id)
                ],
            }
        )

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

        Automatically merges stored budgets from disk with any user-provided
        limits in ``budgets``.

        Args:
            budgets: Optional monthly limits the user gave in this message,
                e.g. {"Food": 400}. Do NOT invent limits; omit if the user did
                not provide any.
        This only suggests; it does not save anything.
        """
        merged = load_budgets(budgets_path)
        if budgets:
            merged.update(budgets)
        return _budget(df, merged)

    @tool
    def save_budget(category: str, monthly_limit: float) -> dict:
        """Save or update a monthly budget limit for a spending category.

        IMPORTANT: Only call this tool when the user explicitly asks to set,
        save, or update a budget limit. Execution will pause for human
        confirmation before writing to disk.

        Args:
            category: Must match one of the dataset's categories (e.g. "Food").
            monthly_limit: Positive monthly budget amount.
        """
        # Case-insensitive match against known dataset categories
        match = next(
            (c for c in known_categories if c.lower() == category.strip().lower()),
            None,
        )
        if match is None:
            raise ValueError(
                f"Unknown category '{category}'. Valid categories: {known_categories}"
            )
        return _save_budget(match, monthly_limit, budgets_path)

    @tool
    def delete_budget(category: str) -> dict:
        """Delete or remove a saved monthly budget limit for a spending category.

        IMPORTANT: Only call this tool when the user explicitly asks to delete
        or remove a saved budget limit. Execution will pause for human
        confirmation before modifying stored budgets.

        Args:
            category: Category whose saved budget should be removed (e.g. "Food").
        """
        match = next(
            (c for c in known_categories if c.lower() == category.strip().lower()),
            category.strip(),
        )
        return _delete_budget(match, budgets_path)

    return [
        get_dataset_overview,
        get_spending_summary,
        compare_spending_periods,
        detect_unusual_expenses,
        get_budget_recommendations,
        save_budget,
        delete_budget,
    ]
