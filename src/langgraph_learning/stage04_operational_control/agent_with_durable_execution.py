"""
Durable Execution: Cached Tasks With Human Approval

=== PROBLEM STATEMENT ===
Marketing and operations agents often pause for human review. Without durable checkpoints,
random seeds and LLM calls rerun after a resume, forcing teams to diff outputs and re-approve
changes.

=== CORE SOLUTION ===
This lesson compiles a four-node LangGraph backed by `SqliteSaver`. Non-deterministic work
is wrapped inside `@task` functions so results are cached in the persistence layer. The graph
raises an interrupt for approval, then resumes without re-invoking the cached tasks.

=== KEY INNOVATION ===
- **Durability Modes**: Streams run with `durability="sync"` so checkpoints flush before
  proceeding.
- **Task Wrappers**: Random IDs and LLM plans live inside `@task`, guaranteeing a single
  execution per run.
- **Interrupt Workflow**: Human reviewers approve the plan by updating state and resuming the
  same thread.

What You'll Learn
1. Configure a SQLite-backed checkpointer that enables durable execution.
2. Wrap side effects and non-deterministic code inside `@task` and retrieve cached results.
3. Pause a workflow with `interrupt`, update thread state, and resume without repeating work.

Lesson Flow
1. Build a campaign-planning graph with cached tasks for ID generation and LLM planning.
2. Stream the graph until an approval interrupt, inspect persisted state, and update the thread.
3. Resume the graph and verify that cached tasks are reused instead of re-running.
"""

from __future__ import annotations

import random
import sqlite3
from pathlib import Path
from pprint import pprint
from typing import Optional

from typing_extensions import NotRequired, TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.func import task
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from src.langgraph_learning.utils import (
    create_llm,
    llm_from_config,
    maybe_enable_langsmith,
    require_llm_provider_api_key,
    save_graph_image,
)


class CampaignState(TypedDict):
    brief: str
    brand: str
    brand_voice: str
    campaign_id: NotRequired[str]
    plan: NotRequired[str]
    approved: NotRequired[bool]
    final_message: NotRequired[str]


LLM_SINGLETON: Optional[ChatOpenAI] = None


def _create_sqlite_saver(db_path: Path) -> SqliteSaver:
    """Return a SQLite-backed saver at the provided location."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path, check_same_thread=False)
    return SqliteSaver(connection)


@task
def _allocate_campaign_id(brand: str) -> str:
    """Generate a random campaign identifier. Cached across resume."""
    ident = random.randint(1000, 9999)
    print(f"[task] Allocated campaign id: {ident}")
    prefix = brand.strip().upper()[:3] or "CMP"
    return f"{prefix}-{ident}"


@task
def _draft_campaign_plan(prompt: str) -> str:
    """Invoke the LLM to produce a campaign plan once per run."""
    if LLM_SINGLETON is None:
        raise RuntimeError("LLM not initialised before drafting campaign plan.")
    print("[task] Calling LLM to draft campaign plan")
    response = LLM_SINGLETON.invoke([HumanMessage(content=prompt)])
    return response.content


def _assemble_durable_graph(llm: ChatOpenAI, saver: SqliteSaver):
    """Return compiled graph that showcases durable execution."""
    global LLM_SINGLETON
    LLM_SINGLETON = llm

    def collect_context(state: CampaignState) -> CampaignState:
        campaign_future = _allocate_campaign_id(state["brand"])
        campaign_id = campaign_future.result()
        return {"campaign_id": campaign_id}

    def draft_plan(state: CampaignState) -> CampaignState:
        prompt = (
            "You are planning a marketing campaign.\n"
            f"Campaign brief: {state['brief']}\n"
            f"Brand voice: {state['brand_voice']}\n"
            f"Campaign id: {state['campaign_id']}\n"
            "Produce a numbered campaign plan with three steps and a launch message."
        )
        plan_future = _draft_campaign_plan(prompt)
        plan = plan_future.result()
        return {"plan": plan}

    def await_approval(state: CampaignState) -> CampaignState:
        if not state.get("approved"):
            print("\n[interrupt] Campaign plan ready for review.")
            raise interrupt(
                "Review the drafted plan. Update the thread with {'approved': True} "
                "or provide edits before resuming."
            )
        return {}

    def finalize(state: CampaignState) -> CampaignState:
        final_message = (
            f"Campaign {state['campaign_id']} approved.\n\nPlan:\n{state['plan']}"
        )
        return {"final_message": final_message}

    builder = StateGraph(CampaignState)
    builder.add_node("collect_context", collect_context)
    builder.add_node("draft_plan", draft_plan)
    builder.add_node("await_approval", await_approval)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "collect_context")
    builder.add_edge("collect_context", "draft_plan")
    builder.add_edge("draft_plan", "await_approval")
    builder.add_edge("await_approval", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=saver)


def build_durable_execution_graph(
    db_path: Path | str = Path("artifacts/durable/state.db"),
):
    llm = create_llm()
    saver = _create_sqlite_saver(Path(db_path))
    graph = _assemble_durable_graph(llm, saver)
    save_graph_image(
        graph, filename="artifacts/durable/agent_with_durable_execution.png"
    )
    return graph


def _thread_config(label: str) -> dict:
    from uuid import uuid4

    return {"configurable": {"thread_id": f"{label}-{uuid4().hex}"}}


def run_until_pause(graph) -> dict:
    thread = _thread_config("durable")
    initial_state: CampaignState = {
        "brief": "Launch a summer cold brew pop-up in downtown San Francisco.",
        "brand": "BrewWave",
        "brand_voice": "Upbeat, community-first, and slightly playful.",
    }
    print("\n--- Running durable campaign planner ---")
    for event in graph.stream(
        initial_state, thread, stream_mode="values", durability="sync"
    ):
        pprint(event)
    state = graph.get_state(thread)
    print("\n--- Persisted state snapshot ---")
    pprint(state.values)
    return thread


def approve_and_resume(graph, thread: dict) -> None:
    print("\n--- Approving cached plan and resuming ---")
    graph.update_state(thread, {"approved": True})
    for event in graph.stream(None, thread, stream_mode="values", durability="sync"):
        pprint(event)
    state = graph.get_state(thread)
    print("\nFinal message:")
    print(state.values.get("final_message", ""))


def main() -> None:
    require_llm_provider_api_key()
    maybe_enable_langsmith()
    graph = build_durable_execution_graph()

    thread = run_until_pause(graph)
    approve_and_resume(graph, thread)


if __name__ == "__main__":
    main()


def studio_graph(config: RunnableConfig | None = None):
    """Studio entry point for the durable execution lesson."""
    llm, overrides = llm_from_config(config)
    db_path = Path(overrides.get("db_path", "artifacts/durable/studio_state.db"))
    saver = _create_sqlite_saver(db_path)
    return _assemble_durable_graph(llm, saver)
