import unittest
from pathlib import Path
import pandas as pd

from tools.analysis import load_transactions, get_spending_summary

class TestAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use the sample CSV provided in the repository
        cls.sample_csv = Path(__file__).resolve().parents[2] / "data" / "sample_transactions.csv"
        # Ensure the file exists
        assert cls.sample_csv.is_file(), f"Sample CSV not found: {cls.sample_csv}"
        cls.df = load_transactions(cls.sample_csv)

    def test_load_transactions_columns(self):
        expected = {"date", "description", "category", "amount"}
        self.assertEqual(set(self.df.columns), expected)

    def test_total_spending(self):
        total, _ = get_spending_summary(self.df)
        # Manual calculation: 3.5+45.2+12.0+60.75+12.5 = 133.95
        self.assertAlmostEqual(total, 133.95, places=2)

    def test_category_totals(self):
        _, cat_totals = get_spending_summary(self.df)
        expected = {
            "Food & Drink": round(3.5 + 45.2, 2),
            "Transport": 12.0,
            "Utilities": 60.75,
            "Entertainment": 12.5,
        }
        self.assertDictEqual(cat_totals, expected)

if __name__ == "__main__":
    unittest.main()
