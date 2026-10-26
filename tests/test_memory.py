"""Phase 4 test: memory + state, using a scripted fake model (no API key needed)."""

import unittest
from pathlib import Path

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from spendwise.agent import SpendWiseState
from spendwise.tools.agent_tools import make_tools

CSV = Path(__file__).resolve().parents[1] / "data" / "sample_transactions.csv"


class FakeToolModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):  # tools are ignored; replies are scripted
        return self


class TestMemoryAndState(unittest.TestCase):
    def test_state_and_thread_memory(self):
        call = AIMessage(
            content="",
            tool_calls=[{
                "name": "compare_spending_periods",
                "id": "c1",
                "args": {
                    "period_a_start": "2025-02-01", "period_a_end": "2025-02-28",
                    "period_b_start": "2025-03-01", "period_b_end": "2025-03-31",
                },
            }],
        )
        replies = iter([call, AIMessage(content="Spending rose."), AIMessage(content="Food.")])
        agent = create_agent(
            FakeToolModel(messages=replies),
            tools=make_tools(CSV),
            state_schema=SpendWiseState,
            checkpointer=InMemorySaver(),
        )
        cfg = {"configurable": {"thread_id": "t1"}}
        agent.invoke({"messages": [("user", "why up?")], "dataset_name": "s.csv"}, cfg)
        agent.invoke({"messages": [("user", "which category?")]}, cfg)

        values = agent.get_state(cfg).values
        self.assertEqual(values["dataset_name"], "s.csv")  # kept across turns
        self.assertEqual(values["last_comparison"]["total_change"], 467.0)  # tool wrote state
        self.assertEqual(len(values["messages"]), 6)  # 2 turns of history remembered

        other = agent.get_state({"configurable": {"thread_id": "t2"}}).values
        self.assertFalse(other)  # a different thread starts empty


if __name__ == "__main__":
    unittest.main()
