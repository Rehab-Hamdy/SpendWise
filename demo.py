"""Simple demo for Phase 1 – load the sample CSV and print a spending summary.

Run with:

    python -m spendwise.demo

The module uses the pure‑Python analysis functions defined in
``spendwise.tools.analysis`` and therefore does not require any LLM.
"""

import json
import sys
from pathlib import Path

# Ensure the package root is on PYTHONPATH when executed directly (e.g. from the
# repository root). When used as ``python -m spendwise.demo`` this is not needed
# because the package is already importable.
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from .tools.analysis import validate_and_summarise

def main() -> None:
    csv_path = Path(__file__).resolve().parent / "data" / "sample_transactions.csv"
    total, category_totals = validate_and_summarise(csv_path)
    summary = {"total_spending": total, "category_totals": category_totals}
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
