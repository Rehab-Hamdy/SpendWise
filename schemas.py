"""Pydantic schemas for SpendWise structured outputs (Expense Reports)."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class UnusualTransactionItem(BaseModel):
    """A single unusually large transaction flagged by analysis."""

    date: str = Field(description="Date of the transaction (YYYY-MM-DD)")
    description: str = Field(description="Merchant or transaction description")
    category: str = Field(description="Spending category")
    amount: float = Field(description="Transaction amount")
    category_median: float = Field(description="Typical (median) spend in this category")


class BudgetRecommendationItem(BaseModel):
    """A budget evaluation and recommendation for a single category."""

    category: str = Field(description="Spending category")
    avg_monthly_spend: float = Field(description="Historical average monthly spend")
    current_budget: Optional[float] = Field(
        default=None, description="Current monthly budget limit if set, otherwise None"
    )
    status: str = Field(
        description="One of: 'over_budget', 'within_budget', or 'no_budget_set'"
    )
    suggested_monthly_budget: float = Field(
        description="Recommended monthly budget target"
    )
    recommendation: str = Field(description="Actionable advice for this category")


class ExpenseReport(BaseModel):
    """Structured expense report combining verified metrics and AI insights."""

    period_covered: str = Field(
        description="Date range covered by the report, e.g. '2025-01-01 to 2025-03-28'"
    )
    total_spending: float = Field(description="Exact total amount spent in the period")
    transaction_count: int = Field(description="Exact number of transactions")
    top_category: str = Field(description="Category with the highest total spend")
    category_breakdown: Dict[str, float] = Field(
        description="Map of category name to total amount spent"
    )
    key_findings: List[str] = Field(
        description="3 to 5 concise, data-backed bullet points summarizing patterns and changes"
    )
    unusual_expenses: List[UnusualTransactionItem] = Field(
        description="List of unusually large transactions flagged in the dataset"
    )
    budget_recommendations: List[BudgetRecommendationItem] = Field(
        description="Actionable budget recommendations per category"
    )
