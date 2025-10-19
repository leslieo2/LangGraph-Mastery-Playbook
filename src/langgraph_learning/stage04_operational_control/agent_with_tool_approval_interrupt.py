"""
Tool-Level Interrupts: Approvals Inside Tools

=== PROBLEM STATEMENT ===
Tools that trigger external actions (payments, deployments, notifications) need human
approval. Static breakpoints pause whole nodes, but they cannot surface tool arguments
for inline review or let operators adjust the payload before proceeding.

=== CORE SOLUTION ===
This lesson binds a tool that calls `interrupt()` from inside its implementation. The
graph captures the pending action, returns it under `__interrupt__`, and resumes only
after the caller responds with `Command(resume=...)`. The tool executes using the
resume payload, so operators can approve, edit, or cancel the action.

=== KEY INNOVATION ===
- **Inline Approval Gate** – `interrupt()` pauses the tool at the exact moment of use.
- **Editable Payloads** – Resume data can override tool arguments before execution.
- **Threaded Persistence** – `MemorySaver` and explicit `thread_id`s keep the run alive.

=== COMPARISON WITH STATIC BREAKPOINTS ===
| agent_with_interruption (static) | This lesson (dynamic tool interrupt) |
|----------------------------------|--------------------------------------|
| Halts before any tool executes   | Stops inside the tool implementation |
| Only offers approve/abort choice | Resume payload can tweak arguments   |
| Uses `interrupt_before=['tools']`| Uses `interrupt()` inside the tool   |

What You'll Learn
1. Bind a LangChain tool that issues `interrupt()` for inline approvals.
2. Resume execution with `Command(resume=...)` and read the returned tool result.
3. Run parallel threads to compare approve vs. reject decisions.

Lesson Flow
1. Build a messages-driven agent with a tool that pauses via `interrupt()`.
2. Launch a run, capture the interrupt payload, and print the pending action.
3. Resume the run twice—once approving with edited values, once rejecting the tool.
"""

from __future__ import annotations

from uuid import uuid4

from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt

from src.langgraph_learning.utils import (
    add,
    create_llm,
    maybe_enable_langsmith,
    multiply,
    require_llm_provider_api_key,
    save_graph_image,
)


@tool
def transfer_funds(amount: float, recipient: str):
    """Pause before transferring funds so an operator can approve or edit."""

    pending = {
        "action": "transfer_funds",
        "amount": amount,
        "recipient": recipient,
        "message": "Approve this transfer? You may edit amount or recipient.",
    }
    decision = interrupt(pending)

    if not isinstance(decision, dict) or not decision.get("approved"):
        return "Transfer cancelled by operator."

    final_amount = decision.get("amount", amount)
    final_recipient = decision.get("recipient", recipient)
    return (
        f"Transferred {final_amount} credits to {final_recipient} "
        f"(approved by operator)."
    )


def build_tool_interrupt_graph():
    """Assemble a tool-enabled assistant whose tool halts for approval."""

    tools = [transfer_funds, add, multiply]
    llm = create_llm()
    llm_with_tools = llm.bind_tools(tools)

    system_prompt = SystemMessage(
        content=(
            "You are a financial assistant. Use tools to perform safe transfers when asked."
        )
    )

    def assistant(state: MessagesState):
        """Call the LLM with accumulated messages plus a system safety prompt."""
        messages = [system_prompt, *state["messages"]]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")
    builder.add_edge("assistant", END)

    graph = builder.compile(checkpointer=MemorySaver())
    save_graph_image(graph, filename="artifacts/agent_with_tool_approval_interrupt.png")
    return graph


def _thread(label: str) -> dict:
    return {"configurable": {"thread_id": f"{label}-{uuid4().hex}"}}


def start_transfer_request(graph, label: str):
    """Kick off a run and return the thread config plus interrupt details."""

    config = _thread(label)
    initial_messages = {
        "messages": [
            HumanMessage(
                content="Transfer 125 credits to alex@example.com and confirm when done."
            )
        ]
    }
    result = graph.invoke(initial_messages, config=config)
    interrupts = result.get("__interrupt__", [])
    if interrupts:
        print(f"[{label}] Interrupt payload:", interrupts[0].value)
    else:
        print(f"[{label}] No interrupt encountered; check tool invocation.")
    return config, interrupts


def approve_transfer(graph, config: dict, edited_amount: float | None = None):
    """Resume execution by approving the transfer, optionally overriding the amount."""

    resume_payload: dict[str, object] = {"approved": True}
    if edited_amount is not None:
        resume_payload["amount"] = edited_amount
    resumed = graph.invoke(Command(resume=resume_payload), config=config)
    final_messages = resumed.get("messages", [])
    if final_messages:
        final_messages[-1].pretty_print()


def reject_transfer(graph, config: dict):
    """Resume execution with a rejection so the tool returns a cancellation message."""

    resumed = graph.invoke(Command(resume={"approved": False}), config=config)
    final_messages = resumed.get("messages", [])
    if final_messages:
        final_messages[-1].pretty_print()


def main() -> None:
    require_llm_provider_api_key()
    maybe_enable_langsmith()
    graph = build_tool_interrupt_graph()

    print("\n--- Approve with edited amount ---")
    approve_config, interrupts = start_transfer_request(graph, "approve-thread")
    if interrupts:
        approve_transfer(graph, approve_config, edited_amount=110)

    print("\n--- Reject the transfer ---")
    reject_config, interrupts = start_transfer_request(graph, "reject-thread")
    if interrupts:
        reject_transfer(graph, reject_config)


if __name__ == "__main__":
    main()
