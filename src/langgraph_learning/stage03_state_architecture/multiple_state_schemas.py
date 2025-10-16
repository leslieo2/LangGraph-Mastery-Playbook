"""This LangGraph workflow experiments with private nodes, shared state, and explicit input/output schemas to control what data each node can access.

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

from src.langgraph_learning.utils import save_graph_image


def demonstrate_private_state() -> None:
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
    graph = builder.compile()

    save_graph_image(graph, filename="artifacts/private_state.png")
    print("Result:", graph.invoke({"foo": 1}))


def demonstrate_single_schema_io() -> None:
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
    graph = builder.compile()

    save_graph_image(graph, filename="artifacts/single_schema_io.png")
    print("Result:", graph.invoke({"question": "hi"}))


def demonstrate_explicit_io_schemas() -> None:
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
    graph = builder.compile()

    save_graph_image(graph, filename="artifacts/explicit_io_schemas.png")
    print("Result:", graph.invoke({"question": "hi"}))


def main() -> None:
    demonstrate_private_state()
    demonstrate_single_schema_io()
    demonstrate_explicit_io_schemas()


if __name__ == "__main__":
    main()
