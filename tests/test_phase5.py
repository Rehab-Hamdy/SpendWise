"""Phase 5 tests: middleware guardrails, tool error handling, structured reports,
transaction evidence accuracy, dataset date overview, and HITL budget save/delete."""

import tempfile
import unittest
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from spendwise.agent import (
    SCOPE_REFUSAL,
    SpendWiseState,
    generate_expense_report,
    get_pending_interrupt,
    input_guard_and_log,
    resume_approval,
    safe_tool_execution,
    validate_and_log_output,
)
from spendwise.schemas import ExpenseReport
from spendwise.tools.agent_tools import make_tools
from spendwise.tools.analysis import (
    compare_spending_periods,
    delete_budget_limit,
    detect_unusual_expenses,
    get_dataset_overview,
    get_spending_summary_for_period,
    load_budgets,
    load_transactions,
    save_budget_limit,
)

CSV = Path(__file__).resolve().parents[1] / "data" / "sample_transactions.csv"


class FakeToolModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class TestPhase5Reliability(unittest.TestCase):
    def test_invalid_csv_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty_csv = Path(tmp) / "empty.csv"
            empty_csv.write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_transactions(empty_csv)

            neg_csv = Path(tmp) / "neg.csv"
            neg_csv.write_text(
                "date,description,category,amount\n2025-01-01,Test,Food,-10\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_transactions(neg_csv)

    def test_input_and_output_middleware_guardrails(self):
        replies = iter([AIMessage(content="Hello, I am ChatGPT from OpenAI!")])
        agent = create_agent(
            FakeToolModel(messages=replies),
            tools=make_tools(CSV),
            state_schema=SpendWiseState,
            middleware=[input_guard_and_log, validate_and_log_output],
            checkpointer=InMemorySaver(),
        )
        cfg = {"configurable": {"thread_id": "guard-test"}}

        # 1) Off-topic Python code request short-circuits before hitting the model
        res1 = agent.invoke(
            {"messages": [("user", "Give me python code for print my name")]}, cfg
        )
        self.assertEqual(res1["messages"][-1].content, SCOPE_REFUSAL)

        # 2) Valid input hits the model, and output middleware sanitizes "ChatGPT" / "OpenAI"
        res2 = agent.invoke({"messages": [("user", "Who are you?")]}, cfg)
        self.assertNotIn("ChatGPT", res2["messages"][-1].content)
        self.assertNotIn("OpenAI", res2["messages"][-1].content)
        self.assertIn("SpendWise", res2["messages"][-1].content)

    def test_tool_error_middleware_catches_bad_args(self):
        bad_call = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_spending_summary",
                    "id": "err1",
                    "args": {"start_date": "2025-03-31", "end_date": "2025-01-01"},
                }
            ],
        )
        replies = iter([bad_call, AIMessage(content="Sorry, those dates were reversed.")])
        agent = create_agent(
            FakeToolModel(messages=replies),
            tools=make_tools(CSV),
            middleware=[safe_tool_execution],
            checkpointer=InMemorySaver(),
        )
        res = agent.invoke(
            {"messages": [("user", "Summary")]},
            {"configurable": {"thread_id": "err-test"}},
        )
        tool_msgs = [m for m in res["messages"] if m.type == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertIn("Tool error", tool_msgs[0].content)

    def test_concert_tickets_transaction_evidence_is_180_not_219(self):
        """Verify March 20 Concert tickets is $180 (not $219 Entertainment total) in tools and agent output."""
        df = load_transactions(CSV)

        # 1) Unusual expenses tool reports Concert tickets on 2025-03-20 as 180.0
        unusual = detect_unusual_expenses(df)["unusual_transactions"]
        concert_unusual = next(t for t in unusual if t["description"] == "Concert tickets")
        self.assertEqual(concert_unusual["date"], "2025-03-20")
        self.assertEqual(concert_unusual["amount"], 180.0)
        self.assertNotEqual(concert_unusual["amount"], 219.0)

        # 2) Period comparison supporting transactions include Concert tickets as 180.0
        comp = compare_spending_periods(
            df, "2025-02-01", "2025-02-28", "2025-03-01", "2025-03-31"
        )
        concert_supp = next(
            t for t in comp["supporting_transactions"] if t["description"] == "Concert tickets"
        )
        self.assertEqual(concert_supp["date"], "2025-03-20")
        self.assertEqual(concert_supp["amount"], 180.0)

        # 3) If an LLM mistakenly writes "$219" for the Concert tickets transaction,
        #    validate_and_log_output middleware corrects it to "$180" using the tool evidence.
        tool_call = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "compare_spending_periods",
                    "id": "cmp1",
                    "args": {
                        "period_a_start": "2025-02-01",
                        "period_a_end": "2025-02-28",
                        "period_b_start": "2025-03-01",
                        "period_b_end": "2025-03-31",
                    },
                }
            ],
        )
        hallucinated_reply = AIMessage(
            content=(
                "Spending increased in March.\n"
                "- 2025-03-20: Concert tickets – $219\n"
                "- 2025-03-13: Anniversary dinner – $210"
            )
        )
        agent = create_agent(
            FakeToolModel(messages=iter([tool_call, hallucinated_reply])),
            tools=make_tools(CSV),
            state_schema=SpendWiseState,
            middleware=[validate_and_log_output],
            checkpointer=InMemorySaver(),
        )
        res = agent.invoke(
            {"messages": [("user", "Why did spending increase in March?")]},
            {"configurable": {"thread_id": "tx-evidence"}},
        )
        final_text = res["messages"][-1].content
        self.assertIn("Concert tickets – $180", final_text)
        self.assertNotIn("Concert tickets – $219", final_text)

    def test_dataset_overview_and_date_range_reporting(self):
        df = load_transactions(CSV)

        # 1) Dedicated get_dataset_overview returns exact CSV bounds (2025-01-01 to 2025-03-28)
        overview = get_dataset_overview(df)
        self.assertEqual(overview["earliest_date"], "2025-01-01")
        self.assertEqual(overview["latest_date"], "2025-03-28")
        self.assertEqual(overview["transaction_count"], 44)
        self.assertEqual(
            overview["categories"],
            ["Entertainment", "Food", "Housing", "Transport", "Utilities"],
        )

        # Also test via the LangChain tool wrapper
        tools_by_name = {t.name: t for t in make_tools(CSV)}
        tool_overview = tools_by_name["get_dataset_overview"].invoke({})
        self.assertEqual(tool_overview["earliest_date"], "2025-01-01")
        self.assertEqual(tool_overview["latest_date"], "2025-03-28")

        # 2) Unbounded get_spending_summary reports actual earliest/latest dates
        full_summary = get_spending_summary_for_period(df)
        self.assertEqual(full_summary["earliest_date"], "2025-01-01")
        self.assertEqual(full_summary["latest_date"], "2025-03-28")
        self.assertEqual(full_summary["total"], 5110.0)

        # 3) Specific requested date range is respected and calculated accurately
        feb_summary = get_spending_summary_for_period(df, "2025-02-01", "2025-02-28")
        self.assertEqual(feb_summary["start"], "2025-02-01")
        self.assertEqual(feb_summary["end"], "2025-02-28")
        self.assertEqual(feb_summary["earliest_date"], "2025-02-01")
        self.assertEqual(feb_summary["latest_date"], "2025-02-27")
        self.assertEqual(feb_summary["transaction_count"], 14)
        self.assertEqual(feb_summary["total"], 1548.0)

    def test_human_in_the_loop_budget_save_and_delete(self):
        with tempfile.TemporaryDirectory() as tmp:
            budgets_file = Path(tmp) / "budgets.json"
            save_call = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "save_budget",
                        "id": "b1",
                        "args": {"category": "Food", "monthly_limit": 350.0},
                    }
                ],
            )
            delete_call = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "delete_budget",
                        "id": "d1",
                        "args": {"category": "Food"},
                    }
                ],
            )
            delete_missing_call = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "delete_budget",
                        "id": "d2",
                        "args": {"category": "Transport"},
                    }
                ],
            )
            replies = iter(
                [
                    save_call,
                    AIMessage(content="Saved Food budget of $350."),
                    save_call,
                    AIMessage(content="Budget change cancelled."),
                    delete_call,
                    AIMessage(content="Deletion cancelled; Food budget kept."),
                    delete_call,
                    AIMessage(content="Deleted Food budget."),
                    delete_missing_call,
                    AIMessage(content="No saved budget found for Transport."),
                ]
            )
            agent = create_agent(
                FakeToolModel(messages=replies),
                tools=make_tools(CSV, budgets_path=budgets_file),
                middleware=[
                    safe_tool_execution,
                    HumanInTheLoopMiddleware(
                        interrupt_on={"save_budget": True, "delete_budget": True}
                    ),
                ],
                checkpointer=InMemorySaver(),
            )

            # 1) Save: triggers interrupt -> approve -> file written
            cfg1 = {"configurable": {"thread_id": "hitl-save-approve"}}
            agent.invoke({"messages": [("user", "Set Food budget to 350")]}, cfg1)
            self.assertIsNotNone(get_pending_interrupt(agent, "hitl-save-approve"))
            self.assertEqual(load_budgets(budgets_file), {})  # not saved before approval
            final_msg = resume_approval(agent, "hitl-save-approve", approved=True)
            self.assertIn("350", final_msg)
            self.assertEqual(load_budgets(budgets_file), {"Food": 350.0})

            # 2) Save reject: file stays {"Food": 350.0}
            cfg2 = {"configurable": {"thread_id": "hitl-save-reject"}}
            agent.invoke({"messages": [("user", "Set Food budget to 500")]}, cfg2)
            self.assertIsNotNone(get_pending_interrupt(agent, "hitl-save-reject"))
            reject_msg = resume_approval(agent, "hitl-save-reject", approved=False)
            self.assertIn("cancelled", reject_msg)
            self.assertEqual(load_budgets(budgets_file), {"Food": 350.0})

            # 3) Delete reject: pauses for approval -> reject -> keeps Food budget unchanged
            cfg3 = {"configurable": {"thread_id": "hitl-del-reject"}}
            agent.invoke({"messages": [("user", "Delete Food budget")]}, cfg3)
            self.assertIsNotNone(get_pending_interrupt(agent, "hitl-del-reject"))
            del_reject_msg = resume_approval(agent, "hitl-del-reject", approved=False)
            self.assertIn("cancelled", del_reject_msg)
            self.assertEqual(load_budgets(budgets_file), {"Food": 350.0})

            # 4) Delete approve: pauses for approval -> approve -> removes Food budget
            cfg4 = {"configurable": {"thread_id": "hitl-del-approve"}}
            agent.invoke({"messages": [("user", "Delete Food budget")]}, cfg4)
            self.assertIsNotNone(get_pending_interrupt(agent, "hitl-del-approve"))
            del_ok_msg = resume_approval(agent, "hitl-del-approve", approved=True)
            self.assertIn("Deleted", del_ok_msg)
            self.assertEqual(load_budgets(budgets_file), {})

            # 5) Delete category with no saved budget: returns deleted=False cleanly
            cfg5 = {"configurable": {"thread_id": "hitl-del-missing"}}
            agent.invoke({"messages": [("user", "Delete Transport budget")]}, cfg5)
            self.assertIsNotNone(get_pending_interrupt(agent, "hitl-del-missing"))
            missing_msg = resume_approval(agent, "hitl-del-missing", approved=True)
            self.assertIn("No saved budget", missing_msg)
            self.assertEqual(load_budgets(budgets_file), {})

    def test_structured_expense_report(self):
        report = generate_expense_report(CSV, model_name="invalid:offline-fallback")
        self.assertIsInstance(report, ExpenseReport)
        self.assertEqual(report.total_spending, 5110.0)
        self.assertEqual(report.transaction_count, 44)
        self.assertEqual(report.top_category, "Housing")
        self.assertGreater(len(report.unusual_expenses), 0)
        self.assertGreater(len(report.budget_recommendations), 0)


if __name__ == "__main__":
    unittest.main()
