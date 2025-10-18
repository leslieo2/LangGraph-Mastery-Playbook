"""
Time Travel: Replay and Fork LangGraph Runs

=== PROBLEM STATEMENT ===
When a conversation diverges, teams often need to rewind to an earlier decision point,
inspect state, and branch into a new trajectory. Without checkpointing, the only option
is to rerun from scratch.

=== CORE SOLUTION ===
This lesson compiles the arithmetic tool agent with `MemorySaver`, runs it once to seed
checkpoints, then shows how to list history, replay a snapshot, and fork a new branch by
mutating state at the saved checkpoint.

=== KEY INNOVATION ===
- **Checkpoint Enumeration**: Surface every stored snapshot along with pending nodes.
- **Replay vs Fork**: Distinguish “read-only replay” from “fork with new inputs”.
- **Thread Configuration**: Demonstrate how LangGraph stores `checkpoint_id` per thread.

=== COMPARISON WITH BREAKPOINT APPROVAL ===
| Breakpoint Control (agent_with_interruption) | Time Travel (this file) |
|---------------------------------------------|-------------------------|
| Pauses before tools to request approval      | Revisits past checkpoints after execution |
| Focus on single linear run                   | Enables branching histories per thread |
| No mutation of saved state                   | Allows new state to branch from a snapshot |

What You'll Learn
1. Capture checkpoints during LangGraph runs and list historical states for inspection.
2. Replay a past checkpoint without mutating state, or fork from it with updated messages.
3. Combine `MemorySaver` with manual state updates to explore alternate execution branches.

Lesson Flow
1. Build an arithmetic agent graph and run it once to populate checkpoint history.
2. Enumerate stored snapshots, choosing one that still has pending nodes.
3. Replay the checkpoint, fork with new inputs, and observe how LangGraph manages divergent threads.
"""

from __future__ import annotations

from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.langgraph_learning.utils import (
    create_llm,
    add,
    divide,
    maybe_enable_langsmith,
    multiply,
    require_llm_provider_api_key,
    save_graph_image,
)


def build_time_travel_graph():
    """Assemble the arithmetic agent and enable checkpointing."""

    tools = [add, multiply, divide]
    llm = create_llm()
    llm_with_tools = llm.bind_tools(tools)
    sys_msg = SystemMessage(
        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
    )

    def assistant(state: MessagesState):
        """Call the tool-enabled model with system prompt + conversation history."""
        return {"messages": [llm_with_tools.invoke([sys_msg, *state["messages"]])]}

    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    graph = builder.compile(checkpointer=MemorySaver())
    save_graph_image(graph, filename="artifacts/agent_with_time_travel.png", xray=True)
    return graph


def _thread_config(prefix: str) -> dict:
    """Helper to mirror doc-style thread configs."""
    return {"configurable": {"thread_id": f"{prefix}-{uuid4().hex}"}}


def _print_messages(event: dict) -> None:
    """Pretty-print the most recent message so the replay feels alive."""
    messages = event.get("messages")
    if not messages:
        print(event)
        return
    last = messages[-1]
    if hasattr(last, "pretty_print"):
        last.pretty_print()
    else:
        print(last)


def run_initial_execution(graph) -> dict:
    """Step 1 from the docs: run the graph once to create checkpoints."""
    config = _thread_config("time-travel")
    input_state = {"messages": [HumanMessage(content="Multiply 2 and 3")]}
    print("\n--- Initial run ---")
    emitted = False
    for event in graph.stream(input_state, config, stream_mode="values"):
        emitted = True
        _print_messages(event)
    if not emitted:
        print("No events produced during the initial run.")
    return config


def list_checkpoints(graph, config: dict):
    """Step 2: list every checkpoint so we can choose one to resume from."""
    history = list(graph.get_state_history(config))
    print("\n--- Stored checkpoints ---")
    for idx, snapshot in enumerate(history):
        checkpoint_id = _extract_checkpoint_id(snapshot)
        next_nodes = getattr(snapshot, "next", ())
        print(f"Checkpoint {idx}: id={checkpoint_id}, next={next_nodes}")
    return history


def _extract_checkpoint_id(snapshot) -> str | None:
    config = (
        snapshot.config if hasattr(snapshot, "config") else snapshot.get("config", {})
    )
    configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
    return configurable.get("checkpoint_id")


def _pick_checkpoint(history):
    """Choose the first snapshot that still has pending work (mirrors doc flow)."""
    for snapshot in history:
        if getattr(snapshot, "next", ()):
            return snapshot
    return history[0] if history else None


def replay_from_snapshot(graph, snapshot) -> None:
    """Step 4 (replay): stream from the checkpoint config without changing state."""
    checkpoint_id = _extract_checkpoint_id(snapshot)
    config = getattr(snapshot, "config", None) or snapshot.get("config", {})
    print(f"\n--- Replaying checkpoint {checkpoint_id} ---")
    emitted = False
    for event in graph.stream(None, config, stream_mode="values"):
        emitted = True
        _print_messages(event)
    if not emitted:
        print(
            "No events were replayed; ensure the checkpoint still contains pending work."
        )


def fork_from_snapshot(graph, snapshot) -> None:
    """Step 3 (optional): tweak the state and continue down a new branch."""
    checkpoint_id = _extract_checkpoint_id(snapshot)
    print(f"\n--- Forking from checkpoint {checkpoint_id} ---")

    snapshot_values = (
        snapshot.values if hasattr(snapshot, "values") else snapshot.get("values", {})
    )
    original_messages = snapshot_values.get("messages", [])
    if not original_messages:
        print("No messages found at checkpoint; skipping fork example.")
        return

    # update_state creates a brand-new checkpoint in the same thread.
    # We append a fresh HumanMessage without reusing the old id so that the new
    # request really asks for 3×3 instead of mutating the previous turn in-place.
    checkpoint_config = getattr(snapshot, "config", None) or snapshot.get("config", {})
    fork_config = graph.update_state(
        checkpoint_config,
        {"messages": [HumanMessage(content="Multiply 3 and 3")]},
        as_node="__start__",
    )

    emitted = False
    for event in graph.stream(None, fork_config, stream_mode="values"):
        emitted = True
        _print_messages(event)
    if not emitted:
        print("No events were emitted after forking; the fork may already be complete.")


def main() -> None:
    require_llm_provider_api_key()
    maybe_enable_langsmith()
    graph = build_time_travel_graph()
    config = run_initial_execution(graph)
    history = list_checkpoints(graph, config)

    if not history:
        print("No checkpoints recorded; aborting replay/fork demo.")
        return

    snapshot = _pick_checkpoint(history)
    if not snapshot:
        print("Unable to locate checkpoint snapshot; aborting replay/fork demo.")
        return

    if not _extract_checkpoint_id(snapshot):
        print("Snapshot did not include checkpoint id; aborting replay/fork demo.")
        return

    replay_from_snapshot(graph, snapshot)
    fork_from_snapshot(graph, snapshot)


if __name__ == "__main__":
    main()
