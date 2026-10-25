import unittest
from pathlib import Path

import pandas as pd

from spendwise.tools.analysis import (
    compare_spending_periods,
    detect_unusual_expenses,
    get_budget_recommendations,
    get_spending_summary,
    get_spending_summary_for_period,
    load_transactions,
)

SAMPLE_CSV = Path(__file__).resolve().parents[1] / "data" / "sample_transactions.csv"


def make_df(rows):
    df = pd.DataFrame(rows, columns=["date", "description", "category", "amount"])
    df["date"] = pd.to_datetime(df["date"])
    return df


# Known inputs: Jan = 180 total, Feb = 300 total
KNOWN = make_df(
    [
        ("2025-01-05", "a", "Food", 100),
        ("2025-01-10", "b", "Food", 50),
        ("2025-01-15", "c", "Transport", 30),
        ("2025-02-05", "d", "Food", 200),
        ("2025-02-10", "e", "Transport", 30),
        ("2025-02-20", "f", "Fun", 70),
    ]
)


class TestPhase1(unittest.TestCase):
    def test_sample_csv_loads(self):
        df = load_transactions(SAMPLE_CSV)
        self.assertEqual(set(df.columns), {"date", "description", "category", "amount"})
        self.assertGreater(len(df), 0)

    def test_summary(self):
        total, cats = get_spending_summary(KNOWN)
        self.assertEqual(total, 480)
        self.assertEqual(cats, {"Food": 350, "Fun": 70, "Transport": 60})


class TestPeriodSummary(unittest.TestCase):
    def test_january_only(self):
        r = get_spending_summary_for_period(KNOWN, "2025-01-01", "2025-01-31")
        self.assertEqual(r["total"], 180)
        self.assertEqual(r["transaction_count"], 3)
        self.assertEqual(r["category_totals"], {"Food": 150, "Transport": 30})


class TestCompare(unittest.TestCase):
    def setUp(self):
        self.r = compare_spending_periods(
            KNOWN, "2025-01-01", "2025-01-31", "2025-02-01", "2025-02-28"
        )

    def test_totals(self):
        self.assertEqual(self.r["total_change"], 120)
        self.assertEqual(self.r["total_pct_change"], 66.67)

    def test_breakdown_sorted_by_change(self):
        b = self.r["category_breakdown"]
        self.assertEqual(b[0]["category"], "Fun")
        self.assertIsNone(b[0]["pct_change"])  # base was zero
        self.assertEqual(b[1]["category"], "Food")
        self.assertEqual(b[1]["change"], 50)

    def test_supporting_transactions(self):
        self.assertEqual(self.r["supporting_transactions"][0]["description"], "f")


class TestUnusual(unittest.TestCase):
    def test_flags_outlier_and_skips_small_categories(self):
        rows = [(f"2025-01-0{i}", "x", "Food", 10) for i in range(1, 5)]
        rows.append(("2025-01-09", "big", "Food", 100))
        rows += [("2025-01-10", "y", "Rare", 500), ("2025-01-11", "z", "Rare", 5)]
        r = detect_unusual_expenses(make_df(rows))
        self.assertEqual(len(r["unusual_transactions"]), 1)
        self.assertEqual(r["unusual_transactions"][0]["description"], "big")
        self.assertEqual(r["skipped_categories"], ["Rare"])

    def test_sample_data_finds_march_spikes(self):
        r = detect_unusual_expenses(load_transactions(SAMPLE_CSV))
        names = {t["description"] for t in r["unusual_transactions"]}
        self.assertIn("Anniversary dinner", names)


class TestBudget(unittest.TestCase):
    def test_recommendations(self):
        r = get_budget_recommendations(KNOWN, budgets={"Food": 150})
        self.assertEqual(r["months_analyzed"], 2)
        by_cat = {x["category"]: x for x in r["recommendations"]}
        self.assertEqual(by_cat["Food"]["avg_monthly_spend"], 175)
        self.assertEqual(by_cat["Food"]["status"], "over_budget")
        self.assertEqual(by_cat["Transport"]["status"], "no_budget_set")
        self.assertEqual(by_cat["Transport"]["suggested_monthly_budget"], 27)


if __name__ == "__main__":
    unittest.main()
