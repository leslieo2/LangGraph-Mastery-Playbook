"""This LangGraph branching mini-agent introduces typed state, conditional routing, and graph visualization around a tiny mood selector.

What You'll Learn
1. Define a minimal `TypedDict` state and state transition nodes in LangGraph.
2. Use conditional edges to control routing based on runtime decisions.
3. Compile and visualize a simple graph before invoking it with example state.

Lesson Flow
1. Describe the `State` schema and implement three node functions.
2. Randomly choose between branches with `decide_mood` and wire conditional edges.
3. Compile the graph, render a PNG artifact, and run a sample invocation.
"""

from __future__ import annotations

import random
from typing import Literal

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from typing_extensions import TypedDict

from src.langgraph_learning.utils import save_graph_image


class State(TypedDict):
    """Shape of the toy conversation state."""

    graph_state: str


def introduce(state: State) -> State:
    print("---Introduce---")
    return {"graph_state": f"{state['graph_state']} I am"}


def happy_path(state: State) -> State:
    print("---Happy---")
    return {"graph_state": f"{state['graph_state']} happy!"}


def sad_path(state: State) -> State:
    print("---Sad---")
    return {"graph_state": f"{state['graph_state']} sad!"}


def decide_mood(_: State) -> Literal["happy", "sad"]:
    """Randomly choose the next node."""
    return random.choice(["happy", "sad"])


def build_graph():
    """Construct a simple branching state graph."""
    graph = StateGraph(State)
    graph.add_node("introduce", introduce)
    graph.add_node("happy", happy_path)
    graph.add_node("sad", sad_path)

    graph.add_edge(START, "introduce")
    graph.add_conditional_edges("introduce", decide_mood)
    graph.add_edge("happy", END)
    graph.add_edge("sad", END)
    return graph.compile()


def visualize(app) -> None:
    """Render the graph to a PNG file."""
    save_graph_image(app, filename="artifacts/simple_state_graph.png")


def run_demo(app) -> None:
    """Invoke the graph with a sample state."""
    result = app.invoke({"graph_state": "Leslie"})
    print("Graph output:", result)


def main() -> None:
    app = build_graph()
    visualize(app)
    run_demo(app)


if __name__ == "__main__":
    main()
