"""This LangGraph agent pauses before tool execution, lets operators approve or resume runs, and mirrors the flow via the hosted LangGraph SDK.

What You'll Learn
1. Use LangGraph breakpoints to pause execution before tool nodes for manual inspection.
2. Resume or cancel runs interactively, simulating approval workflows for tool usage.
3. Stream events via the LangGraph SDK to observe breakpoint behavior in hosted environments.

Lesson Flow
1. Assemble an arithmetic agent graph with `interrupt_before=['tools']` and a memory store.
2. Stream the run until the breakpoint, inspect queued nodes, and optionally continue or abort.
3. Demonstrate CLI approval flow and mirror the same pattern via the LangGraph API client.
"""

from __future__ import annotations

import asyncio
import os
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
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

try:
    from langgraph_sdk import get_client

    HAS_SDK = True
except ImportError:
    HAS_SDK = False


class State(MessagesState):
    """Simple alias so type checkers recognise summary if we add it later."""

    ...


def build_tool_agent_graph_with_breakpoint():
    tools = [add, multiply, divide]

    llm = create_llm()
    llm_with_tools = llm.bind_tools(tools)

    sys_msg = SystemMessage(
        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
    )

    def assistant(state: MessagesState):
        return {"messages": [llm_with_tools.invoke([sys_msg, *state["messages"]])]}

    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    memory = MemorySaver()
    graph = builder.compile(
        interrupt_before=["tools"],
        checkpointer=memory,
    )

    save_graph_image(
        graph, filename="artifacts/agent_with_interruption.png", xray=True
    )
    return graph


def _thread_config(label: str) -> dict:
    return {"configurable": {"thread_id": f"{label}-{uuid4().hex}"}}


def stream_until_interrupt(graph) -> dict:
    thread = _thread_config("bp-demo1")
    initial_input = {"messages": [HumanMessage(content="Multiply 2 and 3")]}
    print("\n--- Streaming until first breakpoint ---")
    for event in graph.stream(initial_input, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()
    state = graph.get_state(thread)
    print("Next node queued:", state.next)
    return thread


def resume_from_breakpoint(graph, thread: dict) -> None:
    print("\n--- Resuming from breakpoint ---")
    # Passing None continues execution from the last checkpoint without new input.
    for event in graph.stream(None, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()


def approval_flow(graph) -> None:
    thread = _thread_config("bp-approval")
    initial_input = {"messages": [HumanMessage(content="Multiply 2 and 3")]}
    print("\n--- Awaiting user approval ---")
    for event in graph.stream(initial_input, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()

    choice = input("Do you want to call the tool? (yes/no): ").strip().lower()
    if choice == "yes":
        resume_from_breakpoint(graph, thread)
    else:
        print("Operation cancelled by user.")


def main() -> None:
    require_llm_provider_api_key()
    maybe_enable_langsmith()
    graph = build_tool_agent_graph_with_breakpoint()

    approval_flow(graph)


if __name__ == "__main__":
    main()
