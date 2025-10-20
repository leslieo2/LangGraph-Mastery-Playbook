"""
Multi-Schema State Management: Controlling Visibility per Node

=== PROBLEM STATEMENT ===
Real workflows often require certain nodes to see private fields while others should only
handle a public subset. Relying on one global schema makes it hard to hide intermediate
values or design limited interfaces for downstream consumers.

=== CORE SOLUTION ===
This lesson walks through three patterns—private node schemas, shared overall schemas,
and explicit input/output contracts—to demonstrate how LangGraph controls what data each
node can read or publish.

=== KEY INNOVATION ===
- **Private State Hops**: Temporarily swap schemas within a node to shield sensitive keys.
- **Selective Field Updates**: Keep a unified schema while nodes touch only the fields
  they own.
- **Explicit IO Contracts**: Use `input_schema` / `output_schema` to expose different
  views externally than what internal nodes carry.

=== COMPARISON WITH SINGLE-SCHEMA GRAPHS ===
| Single Schema Workflows | Multi-Schema Control (this file) |
|-------------------------|----------------------------------|
| Every node sees all keys | Nodes gain scoped visibility |
| Harder to hide internal fields | Private TypedDict snapshots enable isolation |
| External interface matches internal shape | Input/output schemas decouple public contracts |

What You'll Learn
1. Configure LangGraph nodes that operate on different schemas within the same workflow.
2. Restrict or widen visibility by setting explicit input/output schemas on `StateGraph`.
3. Observe how branching nodes behave when state information is partially hidden.

Lesson Flow
1. Demonstrate private node state by introducing a temporary schema and converting back.
2. Build a graph that shares a single overall schema while nodes update only relevant fields.
3. Create an explicit input/output configuration to limit public fields and log the results.
"""

from __future__ import annotations

from typing import Dict

from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph

from langchain_core.runnables import RunnableConfig

from src.langgraph_learning.utils import save_graph_image


def _build_private_state_graph():
    class OverallState(TypedDict):
        foo: int

    class PrivateState(TypedDict):
        baz: int

    def node_1(state: OverallState) -> PrivateState:
        print("---Node 1---")
        return {"baz": state["foo"] + 1}

    def node_2(state: PrivateState) -> OverallState:
        print("---Node 2---")
        return {"foo": state["baz"] + 1}

    builder = StateGraph(OverallState)
    builder.add_node("node_1", node_1)
    builder.add_node("node_2", node_2, input_schema=PrivateState)

    builder.add_edge(START, "node_1")
    builder.add_edge("node_1", "node_2")
    builder.add_edge("node_2", END)
    return builder.compile()


def demonstrate_private_state() -> None:
    graph = _build_private_state_graph()
    save_graph_image(graph, filename="artifacts/private_state.png")
    print("Result:", graph.invoke({"foo": 1}))


def _build_single_schema_graph():
    # Demonstrates a graph where the same overall schema is used for input, internal
    # updates, and output—nodes update only the fields they care about, but all keys
    # remain visible at the graph boundary.
    class OverallState(TypedDict):
        question: str
        answer: str
        notes: str

    def thinking_node(state: OverallState) -> Dict[str, str]:
        print("---Thinking node---")
        return {"answer": "bye", "notes": "... his name is Leslie"}

    def answer_node(state: OverallState) -> Dict[str, str]:
        print("---Answer node---")
        return {"answer": "bye Leslie"}

    builder = StateGraph(OverallState)
    builder.add_node("thinking_node", thinking_node)
    builder.add_node("answer_node", answer_node)
    builder.add_edge(START, "thinking_node")
    builder.add_edge("thinking_node", "answer_node")
    builder.add_edge("answer_node", END)
    return builder.compile()


def demonstrate_single_schema_io() -> None:
    graph = _build_single_schema_graph()
    save_graph_image(graph, filename="artifacts/single_schema_io.png")
    print("Result:", graph.invoke({"question": "hi"}))


def _build_explicit_io_graph():
    # Demonstrates how to keep internal state rich while constraining the public
    # interface via dedicated input and output schemas.
    class InputState(TypedDict):
        question: str

    class OutputState(TypedDict):
        answer: str

    class OverallState(TypedDict):
        question: str
        answer: str
        notes: str

    def thinking_node(state: InputState) -> Dict[str, str]:
        print("---Thinking node (filtered input)---")
        return {"answer": "bye", "notes": "... his name is Leslie"}

    def answer_node(state: OverallState) -> OutputState:
        print("---Answer node (filtered output)---")
        return {"answer": "bye Leslie"}

    builder = StateGraph(
        OverallState, input_schema=InputState, output_schema=OutputState
    )
    builder.add_node("thinking_node", thinking_node)
    builder.add_node("answer_node", answer_node)
    builder.add_edge(START, "thinking_node")
    builder.add_edge("thinking_node", "answer_node")
    builder.add_edge("answer_node", END)
    return builder.compile()


def demonstrate_explicit_io_schemas() -> None:
    graph = _build_explicit_io_graph()
    save_graph_image(graph, filename="artifacts/agent_with_multiple_state.png")
    print("Result:", graph.invoke({"question": "hi"}))


def main() -> None:
    demonstrate_private_state()
    demonstrate_single_schema_io()
    demonstrate_explicit_io_schemas()


if __name__ == "__main__":
    main()


def studio_graph(config: RunnableConfig | None = None):
    """Studio entry point for exploring multiple state schema patterns."""
    variant = (config or {}).get("configurable", {}).get("variant") if config else None
    if variant == "private":
        return _build_private_state_graph()
    if variant == "single":
        return _build_single_schema_graph()
    # Default to explicit IO schema example.
    return _build_explicit_io_graph()
