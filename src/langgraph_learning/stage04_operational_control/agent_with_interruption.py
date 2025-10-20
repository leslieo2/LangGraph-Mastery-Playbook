"""
Tool Breakpoints: Manual Approval Before Execution

=== PROBLEM STATEMENT ===
Tool-enabled agents can execute side-effecting operations. Without guardrails, they may
call external APIs or mutate data before an operator approves the action.

=== CORE SOLUTION ===
This lesson compiles an arithmetic tool agent with `interrupt_before=['tools']`, forcing
LangGraph to pause immediately before any tool call. Operators can inspect queued tasks,
resume, or abort—mirroring approval workflows.

=== KEY INNOVATION ===
- **Graph-Level Breakpoints**: Pause the run before specific nodes fire.
- **Interactive Resume**: Demonstrates both CLI prompts and programmatic resumes.
- **Hosted Parity**: Mirrors the same approval dance through the LangGraph SDK client.

=== COMPARISON WITH DYNAMIC INTERRUPTS ===
| Dynamic Interrupts (agent_with_dynamic_interruption) | Tool Breakpoints (this file) |
|------------------------------------------------------|------------------------------|
| Nodes raise `interrupt` from inside execution        | Graph pauses before node execution |
| Requires catching errors during runtime              | Pre-emptively halts tools for review |
| Focus on repairing state mid-node                    | Focus on approving or cancelling side effects |

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
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.langgraph_learning.utils import (
    add,
    create_llm,
    llm_from_config,
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


TOOLS = [add, multiply, divide]
SYSTEM_MESSAGE = SystemMessage(
    content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
)


def _assemble_tool_agent_graph(
    llm, memory: MemorySaver, *, interrupt_before: list[str]
):
    """Return compiled tool agent graph using supplied dependencies."""
    llm_with_tools = llm.bind_tools(TOOLS)

    def assistant(state: MessagesState):
        return {
            "messages": [llm_with_tools.invoke([SYSTEM_MESSAGE, *state["messages"]])]
        }

    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    return builder.compile(
        interrupt_before=interrupt_before,
        checkpointer=memory,
    )


def build_tool_agent_graph_with_breakpoint():
    llm = create_llm()
    memory = MemorySaver()
    graph = _assemble_tool_agent_graph(llm, memory, interrupt_before=["tools"])

    save_graph_image(graph, filename="artifacts/agent_with_interruption.png", xray=True)
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


def studio_graph(config: RunnableConfig | None = None):
    """Studio entry point for the breakpoint-enabled tool agent."""
    llm, overrides = llm_from_config(config)
    interrupt_before = overrides.get("interrupt_before") or ["tools"]
    memory = MemorySaver()
    return _assemble_tool_agent_graph(
        llm,
        memory,
        interrupt_before=list(interrupt_before),
    )
