"""This LangGraph guardrail demo interrupts mid-run when input fails a length check, then walks through inspecting, retrying, and repairing state.

What You'll Learn
1. Trigger dynamic interrupts inside LangGraph nodes using `langgraph.types.interrupt`.
2. Inspect pending tasks, retry without changes, and update state to satisfy guards.

Lesson Flow
1. Build a three-step graph where the middle node interrupts on long input strings.
2. Stream execution until the interrupt, examine the stored state, and attempt a no-op resume.
3. Modify the state to clear the guard, continue the run locally.
"""

from __future__ import annotations

import asyncio
import os
from uuid import uuid4

from langgraph.types import interrupt
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.langgraph_learning.utils import save_graph_image

try:
    from langgraph_sdk import get_client

    HAS_SDK = True
except ImportError:
    HAS_SDK = False


class State(TypedDict):
    input: str


def build_dynamic_breakpoint_graph():
    def step_1(state: State) -> State:
        print("---Step 1---")
        return state

    def step_2(state: State) -> State:
        if len(state["input"]) > 5:
            raise interrupt(
                f"Received input that is longer than 5 characters: {state['input']}"
            )
        print("---Step 2---")
        return state

    def step_3(state: State) -> State:
        print("---Step 3---")
        return state

    builder = StateGraph(State)
    builder.add_node("step_1", step_1)
    builder.add_node("step_2", step_2)
    builder.add_node("step_3", step_3)

    builder.add_edge(START, "step_1")
    builder.add_edge("step_1", "step_2")
    builder.add_edge("step_2", "step_3")
    builder.add_edge("step_3", END)

    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    save_graph_image(graph, filename="artifacts/debugging/agent_with_dynamic_interruption.png")
    return graph


def _thread_config(label: str) -> dict:
    return {"configurable": {"thread_id": f"{label}-{uuid4().hex}"}}


def run_until_interrupt(graph) -> dict:
    input_state = {"input": "hello world"}
    thread = _thread_config("dynamic")
    print("\n--- Running until dynamic interrupt ---")
    for event in graph.stream(input_state, thread, stream_mode="values"):
        print(event)
    state = graph.get_state(thread)
    print("Next node queued:", state.next)
    print("Tasks:", state.tasks)
    return thread


def retry_without_change(graph, thread: dict) -> None:
    print("\n--- Attempting to resume without updating state ---")
    for event in graph.stream(None, thread, stream_mode="values"):
        print(event)
    state = graph.get_state(thread)
    print("Next node still queued:", state.next)


def update_state_and_resume(graph, thread: dict) -> None:
    print("\n--- Updating state to satisfy guard ---")
    graph.update_state(thread, {"input": "hi"})
    for event in graph.stream(None, thread, stream_mode="values"):
        print(event)


def main() -> None:
    graph = build_dynamic_breakpoint_graph()
    thread = run_until_interrupt(graph)
    retry_without_change(graph, thread)
    update_state_and_resume(graph, thread)


if __name__ == "__main__":
    main()
