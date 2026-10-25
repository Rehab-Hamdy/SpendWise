"""SpendWise agent: model + system prompt + tools.

Run an interactive chat in the terminal with:

    python -m spendwise.agent
    python -m spendwise.agent path/to/your.csv
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from .tools.agent_tools import make_tools
from .tools.analysis import load_transactions

load_dotenv()  # reads API keys + MODEL_NAME from .env

DEFAULT_CSV = Path(__file__).resolve().parent / "data" / "sample_transactions.csv"
# "provider:model" - see .env.example for free options
DEFAULT_MODEL = "groq:llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are SpendWise, a personal expense analysis assistant.

Dataset: transactions from {first} to {last}. Categories: {categories}.
Today's date is not meaningful here: treat "this month" / "latest month" as the
last month in the dataset, and "last month" as the month before it.

Rules:
- NEVER calculate or guess numbers yourself. Every figure you state must come
  from a tool result. If you need a number, call a tool.
- Pick the tool that matches the question; call several if needed (e.g. compare
  periods, then look at supporting transactions).
- For "why did spending change?" questions, use compare_spending_periods and
  explain using the category breakdown and supporting transactions.
- Use full calendar-month dates (e.g. 2025-03-01 to 2025-03-31) when comparing months.
- If the data cannot answer the question, say so plainly. Do not invent data.
- You only analyze and suggest. You cannot save or change budgets or transactions.
- Be concise: lead with the answer, then 2-4 bullet points of evidence.
"""


def build_agent(csv_path: str | Path = DEFAULT_CSV, model_name: str | None = None):
    """Create the agent for one CSV file."""
    model_name = model_name or os.getenv("MODEL_NAME", DEFAULT_MODEL)
    model = init_chat_model(model_name, temperature=0)

    df = load_transactions(csv_path)
    prompt = SYSTEM_PROMPT.format(
        first=df["date"].min().date(),
        last=df["date"].max().date(),
        categories=", ".join(sorted(df["category"].unique())),
    )
    return create_agent(model, tools=make_tools(csv_path), system_prompt=prompt)


def ask(agent, question: str, verbose: bool = True) -> str:
    """Send one question; optionally print which tools were called."""
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    if verbose:
        for msg in result["messages"]:
            for call in getattr(msg, "tool_calls", None) or []:
                print(f"  [tool] {call['name']}({call['args']})")
    return result["messages"][-1].content


def main() -> None:
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    agent = build_agent(csv_path)
    print("SpendWise ready. Type a question (or 'exit').")
    while (q := input("\nYou: ").strip()).lower() not in {"exit", "quit"}:
        if q:
            print(f"\nSpendWise: {ask(agent, q)}")


if __name__ == "__main__":
    main()
