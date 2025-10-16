"""This LangGraph reducer sandbox demonstrates overwrite conflicts, custom list merging, and direct message manipulation inside compiled workflows.

What You'll Learn
1. Explore LangGraph's reducer semantics for overwriting, branching conflicts, and list appends.
2. Implement custom reducer functions that safely merge list state with nullable inputs.
3. Manipulate message history directly via `add_messages`, including deletion instructions.

Lesson Flow
1. Walk through default overwrite behavior and inspect the resulting state output.
2. Trigger a branching conflict, then switch to append reducers and custom merge logic.
3. Demonstrate message append/overwrite/removal, printing intermediate results along the way.
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Dict

from typing_extensions import TypedDict

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.errors import InvalidUpdateError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from src.langgraph_learning.utils import save_graph_image


class OverwriteState(TypedDict):
    foo: int


class BranchingState(TypedDict):
    foo: int


class AppendReducerState(TypedDict):
    foo: Annotated[list[int], add]


def reduce_list(left: list[int] | None, right: list[int] | None) -> list[int]:
    """Safely concatenate two lists while tolerating None inputs."""
    left = left or []
    right = right or []
    return left + right


def render_graph(graph, filename: str) -> None:
    """Persist the PNG for the compiled graph."""
    save_graph_image(graph, filename=filename)


def demonstrate_default_overwrite() -> None:
    def node_increment(state: OverwriteState) -> Dict[str, int]:
        print("---Node increment---")
        return {"foo": state["foo"] + 1}

    builder = StateGraph(OverwriteState)
    builder.add_node("node_increment", node_increment)
    builder.add_edge(START, "node_increment")
    builder.add_edge("node_increment", END)
    graph = builder.compile()

    render_graph(graph, "artifacts/default_overwrite.png")
    result = graph.invoke({"foo": 1})
    print("Result:", result)


def demonstrate_branching_conflict() -> None:
    def node_1(state: BranchingState) -> Dict[str, int]:
        print("---Node 1---")
        return {"foo": state["foo"] + 1}

    def node_2(state: BranchingState) -> Dict[str, int]:
        print("---Node 2---")
        return {"foo": state["foo"] + 1}

    def node_3(state: BranchingState) -> Dict[str, int]:
        print("---Node 3---")
        return {"foo": state["foo"] + 1}

    builder = StateGraph(BranchingState)
    builder.add_node("node_1", node_1)
    builder.add_node("node_2", node_2)
    builder.add_node("node_3", node_3)
    builder.add_edge(START, "node_1")
    builder.add_edge("node_1", "node_2")
    builder.add_edge("node_1", "node_3")
    builder.add_edge("node_2", END)
    builder.add_edge("node_3", END)
    graph = builder.compile()

    render_graph(graph, "artifacts/branching_conflict.png")
    try:
        print(graph.invoke({"foo": 1}))
    except InvalidUpdateError as exc:
        print("InvalidUpdateError occurred:", exc)


def demonstrate_reducer_append() -> None:
    def node_1(state: AppendReducerState) -> Dict[str, list[int]]:
        print("---Node 1---")
        return {"foo": [state["foo"][-1] + 1]}

    def node_2(state: AppendReducerState) -> Dict[str, list[int]]:
        print("---Node 2---")
        return {"foo": [state["foo"][-1] + 1]}

    def node_3(state: AppendReducerState) -> Dict[str, list[int]]:
        print("---Node 3---")
        return {"foo": [state["foo"][-1] + 1]}

    builder = StateGraph(AppendReducerState)
    builder.add_node("node_1", node_1)
    builder.add_node("node_2", node_2)
    builder.add_node("node_3", node_3)
    builder.add_edge(START, "node_1")
    builder.add_edge("node_1", "node_2")
    builder.add_edge("node_1", "node_3")
    builder.add_edge("node_2", END)
    builder.add_edge("node_3", END)
    graph = builder.compile()

    render_graph(graph, "artifacts/reducer_append.png")
    print(graph.invoke({"foo": [1]}))
    try:
        graph.invoke({"foo": None})  # type: ignore[arg-type]
    except TypeError as exc:
        print("TypeError occurred:", exc)


def demonstrate_messages_reducer() -> None:
    initial_messages = [
        AIMessage(content="Hello! How can I assist you?", name="Model"),
        HumanMessage(
            content="I'm looking for information on marine biology.", name="Leslie"
        ),
    ]
    new_message = AIMessage(
        content="Sure, I can help with that. What specifically are you interested in?",
        name="Model",
    )

    print("\n--- Messages: append ---")
    print(add_messages(initial_messages, new_message))

    overwrite_source = [
        AIMessage(content="Hello! How can I assist you?", name="Model", id="1"),
        HumanMessage(
            content="I'm looking for information on marine biology.",
            name="Leslie",
            id="2",
        ),
    ]
    overwrite_message = HumanMessage(
        content="I'm looking for information on whales, specifically",
        name="Leslie",
        id="2",
    )

    print("\n--- Messages: overwrite by id ---")
    print(add_messages(overwrite_source, overwrite_message))

    messages = [
        AIMessage("Hi.", name="Bot", id="1"),
        HumanMessage("Hi.", name="Leslie", id="2"),
        AIMessage(
            "So you said you were researching ocean mammals?", name="Bot", id="3"
        ),
        HumanMessage(
            "Yes, I know about whales. But what others should I learn about?",
            name="Leslie",
            id="4",
        ),
    ]
    delete_messages = [RemoveMessage(id=m.id) for m in messages[:2]]

    print("\n--- Messages: removal ---")
    print("Delete instructions:", delete_messages)
    print(add_messages(messages, delete_messages))


def main() -> None:
    demonstrate_default_overwrite()
    demonstrate_branching_conflict()
    demonstrate_reducer_append()
    demonstrate_messages_reducer()


if __name__ == "__main__":
    main()
