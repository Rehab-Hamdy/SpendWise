"""SpendWise Streamlit Web Application (Phase 6).

Run from the repository root with:
    python -m streamlit run spendwise/app.py
"""

from __future__ import annotations

import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# Ensure the repository root is on sys.path whether run via `streamlit run spendwise/app.py`
# or `python -m streamlit run spendwise/app.py`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spendwise.agent import (
    DEFAULT_CSV,
    build_agent,
    configure_langsmith,
    generate_expense_report,
    get_pending_interrupt,
    make_run_config,
    resume_approval,
)
from spendwise.tools.analysis import (
    DEFAULT_BUDGETS_PATH,
    detect_unusual_expenses,
    get_budget_recommendations,
    get_dataset_overview,
    get_spending_summary_for_period,
    load_budgets,
    load_transactions,
)


def send_chat_message(
    agent,
    question: str,
    thread_id: str,
    dataset_name: str,
) -> dict[str, Any]:
    """Invoke the SpendWise agent for one chat turn and return reply, tool calls, and HITL state."""
    config = make_run_config(
        thread_id,
        run_name="SpendWiseStreamlitTurn",
        dataset_name=dataset_name,
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}], "dataset_name": dataset_name},
        config,
    )
    messages = result.get("messages", [])
    last_human = max(
        (i for i, m in enumerate(messages) if getattr(m, "type", None) == "human"),
        default=0,
    )
    tool_calls = []
    for msg in messages[last_human:]:
        for call in getattr(msg, "tool_calls", None) or []:
            tool_calls.append({"name": call["name"], "args": call["args"]})

    pending = get_pending_interrupt(agent, thread_id)
    reply = messages[-1].content if messages and not pending else ""
    return {
        "reply": reply,
        "tool_calls": tool_calls,
        "pending_interrupt": pending,
    }


def _init_session(csv_path: Path, dataset_name: str) -> None:
    """Initialize or reset the Streamlit session state for the selected CSV dataset."""
    st.session_state.csv_path = str(csv_path)
    st.session_state.dataset_name = dataset_name
    st.session_state.agent = build_agent(csv_path)
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.chat_history = []
    st.session_state.report = None


def main() -> None:
    st.set_page_config(
        page_title="SpendWise — Agentic Personal Expense Investigator",
        page_icon="💳",
        layout="wide",
    )

    st.title("💳 SpendWise — Agentic Personal Expense Investigator")
    st.caption(
        "Upload a transactions CSV, explore verified spending metrics, and investigate your expenses with a LangChain tool-calling agent."
    )

    # ------------------------------- Sidebar: Data Source & Session Controls
    with st.sidebar:
        st.header("📂 Data Source")
        uploaded_file = st.file_uploader(
            "Upload transactions CSV",
            type=["csv"],
            help="Required columns: date, description, category, amount",
        )

        target_csv = DEFAULT_CSV
        target_name = DEFAULT_CSV.name

        if uploaded_file is not None:
            tmp_dir = Path(tempfile.gettempdir()) / "spendwise_uploads"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            candidate_path = tmp_dir / uploaded_file.name
            candidate_path.write_bytes(uploaded_file.getvalue())
            try:
                load_transactions(candidate_path)
                target_csv = candidate_path
                target_name = uploaded_file.name
                st.success(f"Loaded `{target_name}`")
            except Exception as exc:
                st.error(f"Invalid CSV file: {exc}")
                st.info("Falling back to sample transactions dataset.")

        if (
            "agent" not in st.session_state
            or st.session_state.get("csv_path") != str(target_csv)
        ):
            _init_session(target_csv, target_name)

        st.divider()
        st.subheader("💬 Session Controls")
        st.write(f"**Active dataset:** `{st.session_state.dataset_name}`")
        st.write(f"**Thread ID:** `{st.session_state.thread_id[:8]}`")
        ls_info = configure_langsmith()
        if ls_info["enabled"]:
            st.write(f"**LangSmith:** 🟢 Active (`{ls_info['project']}`)")
        else:
            st.write("**LangSmith:** ⚪ Disabled")

        if st.button("🔄 New Conversation Thread", use_container_width=True):
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.chat_history = []
            st.rerun()

        st.divider()
        st.subheader("💰 Saved Monthly Budgets")
        saved_budgets = load_budgets(DEFAULT_BUDGETS_PATH)
        if saved_budgets:
            for cat, limit in saved_budgets.items():
                st.write(f"- **{cat}:** ${limit:,.2f} / month")
        else:
            st.caption("No saved budgets yet. Ask the agent to save a budget in chat!")

    # Load validated DataFrame and pure-Python metrics
    df = load_transactions(st.session_state.csv_path)
    overview = get_dataset_overview(df)
    summary = get_spending_summary_for_period(df)
    unusual = detect_unusual_expenses(df)
    budget_recs = get_budget_recommendations(df, saved_budgets)

    # ------------------------------------------------- Top Key Metrics Cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Spending", f"${overview['total_spending']:,.2f}")
    col2.metric("Transactions", f"{overview['transaction_count']}")
    top_cat_name = next(iter(summary["category_totals"]), "N/A")
    top_cat_amt = summary["category_totals"].get(top_cat_name, 0.0)
    col3.metric("Top Category", f"{top_cat_name} (${top_cat_amt:,.2f})")
    col4.metric(
        "Date Range",
        f"{overview['earliest_date']} → {overview['latest_date']}",
    )

    # ------------------------------------------------------------------ Tabs
    tab_chat, tab_charts, tab_insights, tab_report = st.tabs(
        [
            "💬 Ask SpendWise",
            "📊 Spending Charts",
            "🔍 Unusual Expenses & Budgets",
            "📑 Structured Expense Report",
        ]
    )

    # ===================================================== Tab 1: Agent Chat
    with tab_chat:
        st.subheader("Investigate Your Spending")
        st.caption(
            "Try asking: *'Why did my spending increase in March compared to February?'*, "
            "*'Are there any unusual expenses?'*, or *'Save a monthly budget of 350 for Food'*."
        )

        # Render conversation history
        for entry in st.session_state.chat_history:
            with st.chat_message(entry["role"]):
                if entry.get("tool_calls"):
                    with st.expander("🛠️ Tools executed", expanded=False):
                        for tc in entry["tool_calls"]:
                            st.code(f"{tc['name']}({tc['args']})", language="python")
                st.markdown(entry["content"])

        # Check if Human-in-the-Loop approval is waiting on this thread
        pending = get_pending_interrupt(
            st.session_state.agent, st.session_state.thread_id
        )
        if pending:
            st.warning("⏸️ **Confirmation Required Before Modifying Budgets**")
            for act in pending.get("action_requests", []):
                st.code(act.get("description", str(act)))
            btn_col1, btn_col2, _ = st.columns([1, 1, 4])
            if btn_col1.button("✅ Approve", type="primary", use_container_width=True):
                try:
                    with st.spinner("Applying change and continuing..."):
                        reply = resume_approval(
                            st.session_state.agent,
                            st.session_state.thread_id,
                            approved=True,
                        )
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": reply, "tool_calls": []}
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(
                        f"⚠️ Provider rate limit or error: {exc}\n\n"
                        "Wait ~15 seconds and click **Approve** again (your confirmation state is still saved)."
                    )
            if btn_col2.button("❌ Reject", use_container_width=True):
                try:
                    with st.spinner("Cancelling change..."):
                        reply = resume_approval(
                            st.session_state.agent,
                            st.session_state.thread_id,
                            approved=False,
                        )
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": reply, "tool_calls": []}
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(
                        f"⚠️ Provider rate limit or error: {exc}\n\n"
                        "Wait ~15 seconds and click **Reject** again."
                    )

        prompt = st.chat_input(
            "Ask a question about your expenses...",
            disabled=bool(pending),
        )
        if prompt:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            try:
                with st.spinner("Analyzing transactions..."):
                    turn = send_chat_message(
                        st.session_state.agent,
                        prompt,
                        st.session_state.thread_id,
                        st.session_state.dataset_name,
                    )
                if turn["pending_interrupt"]:
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": "*Waiting for your confirmation to modify stored budgets...*",
                            "tool_calls": turn["tool_calls"],
                        }
                    )
                else:
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": turn["reply"],
                            "tool_calls": turn["tool_calls"],
                        }
                    )
                st.rerun()
            except Exception as exc:
                st.session_state.chat_history.pop()
                st.error(
                    f"⚠️ Free-tier API rate limit reached: {exc}\n\n"
                    "Wait ~15 seconds and try again, or switch `MODEL_NAME` in `spendwise/.env` to "
                    "`groq:llama-3.3-70b-versatile` (12,000 TPM)."
                )

    # ================================================ Tab 2: Spending Charts
    with tab_charts:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Spending by Category")
            cat_df = pd.DataFrame(
                list(summary["category_totals"].items()),
                columns=["Category", "Total ($)"],
            ).set_index("Category")
            st.bar_chart(cat_df)

        with c2:
            st.subheader("Monthly Spending Trend")
            monthly_df = (
                df.assign(month=df["date"].dt.to_period("M").astype(str))
                .groupby(["month", "category"])["amount"]
                .sum()
                .unstack(fill_value=0)
            )
            st.bar_chart(monthly_df)

        st.subheader("Top Transactions")
        st.dataframe(
            pd.DataFrame(summary["top_transactions"]),
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("View All Transactions"):
            display_df = df.copy()
            display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    # =============================== Tab 3: Unusual Expenses & Budget Advice
    with tab_insights:
        st.subheader("🚨 Unusual Transactions (IQR Outlier Detection)")
        if unusual["unusual_transactions"]:
            st.dataframe(
                pd.DataFrame(unusual["unusual_transactions"]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No unusual transactions detected.")

        st.subheader("💡 Data-Driven Budget Recommendations")
        st.caption(
            f"Based on {budget_recs['months_analyzed']} month(s) of transaction history."
        )
        st.dataframe(
            pd.DataFrame(budget_recs["recommendations"]),
            use_container_width=True,
            hide_index=True,
        )

    # ======================================= Tab 4: Structured Expense Report
    with tab_report:
        st.subheader("📑 Structured Expense Report")
        st.caption(
            "Generates a Pydantic-validated `ExpenseReport` combining exact Python calculations with AI key findings."
        )
        if st.button("✨ Generate Expense Report", type="primary"):
            with st.spinner("Generating structured report..."):
                st.session_state.report = generate_expense_report(
                    st.session_state.csv_path,
                    budgets_path=DEFAULT_BUDGETS_PATH,
                )

        if st.session_state.report is not None:
            rep = st.session_state.report
            st.markdown(f"**Period Covered:** `{rep.period_covered}`")
            st.markdown("### Key Findings")
            for finding in rep.key_findings:
                st.markdown(f"- {finding}")

            with st.expander("View Raw Validated JSON"):
                st.json(rep.model_dump())


if __name__ == "__main__":
    main()
