"""SpendWise agent: model + system prompt + tools.

Run an interactive chat in the terminal with:

    python -m spendwise.agent
    python -m spendwise.agent path/to/your.csv
"""

import os
import sys
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

from .tools.agent_tools import make_tools
from .tools.analysis import load_transactions


class SpendWiseState(AgentState):
    """Session state. `messages` (inherited) holds the conversation; we add:

    - dataset_name: which CSV this session is analysing
    - last_comparison: result of the latest period comparison, written by the
      compare_spending_periods tool, so follow-ups ("which category drove
      that?") have the exact numbers available.
    """

    dataset_name: str
    last_comparison: dict[str, Any]


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
- This is an ongoing conversation. For follow-ups ("which category contributed
  most?"), reuse the earlier tool results in the conversation instead of asking
  the user to repeat themselves; call a tool again only if new data is needed.
"""


def build_agent(csv_path: str | Path = DEFAULT_CSV, model_name: str | None = None):
    """Create the agent for one CSV file."""
    model_name = model_name or os.getenv("MODEL_NAME", DEFAULT_MODEL)
    model = init_chat_model(model_name, temperature=0.1)

    df = load_transactions(csv_path)
    prompt = SYSTEM_PROMPT.format(
        first=df["date"].min().date(),
        last=df["date"].max().date(),
        categories=", ".join(sorted(df["category"].unique())),
    )
    return create_agent(
        model,
        tools=make_tools(csv_path),
        system_prompt=prompt,
        state_schema=SpendWiseState,
        # The checkpointer saves state after every step, keyed by thread_id.
        # InMemorySaver lives in RAM: memory is lost when the program exits.
        checkpointer=InMemorySaver(),
    )


def ask(
    agent,
    question: str,
    thread_id: str,
    dataset_name: str = "sample_transactions.csv",
    verbose: bool = True,
) -> str:
    """Send one question within a conversation (thread)."""
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}], "dataset_name": dataset_name},
        config,
    )
    if verbose:
        # The result holds the WHOLE history now; show only this turn's tool calls.
        last_human = max(i for i, m in enumerate(result["messages"]) if m.type == "human")
        for msg in result["messages"][last_human:]:
            for call in getattr(msg, "tool_calls", None) or []:
                print(f"  [tool] {call['name']}({call['args']})")
    return result["messages"][-1].content


def main() -> None:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    agent = build_agent(csv_path)
    thread_id = str(uuid.uuid4())  # new id = new conversation
    config = {"configurable": {"thread_id": thread_id}}
    print(f"SpendWise ready (thread {thread_id[:8]}). Commands: /state, /new, exit")
    while (q := input("\nYou: ").strip()).lower() not in {"exit", "quit"}:
        if q == "/state":  # peek at what the checkpointer stored
            values = agent.get_state(config).values
            print(f"  messages stored: {len(values.get('messages', []))}")
            print(f"  dataset_name: {values.get('dataset_name')}")
            lc = values.get("last_comparison")
            print("  last_comparison:", f"{lc['period_a']} -> {lc['period_b']}" if lc else None)
        elif q == "/new":  # different thread = fresh memory
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}
            print(f"  new conversation (thread {thread_id[:8]})")
        elif q:
            print(f"\nSpendWise: {ask(agent, q, thread_id, csv_path.name)}")


if __name__ == "__main__":
    main()
