"""
Advanced Memory Management: Summaries, Resets, and Checkpoint Ops

=== PROBLEM STATEMENT ===
Short-term memory grows without bound, while operators still need visibility
into stored checkpoints for audits. Manual pruning is error-prone, and most
agents lack tooling to summarize, purge, or inspect state safely.

=== CORE SOLUTION ===
This lesson combines LangMem’s `SummarizationNode`, full-history wipes via
`REMOVE_ALL_MESSAGES`, and production checkpoint APIs to monitor and manage
conversation state.

=== KEY INNOVATION ===
- **Reusable Summaries**: `SummarizationNode` maintains a rolling abstract so
  downstream nodes work with concise context.
- **Controlled Resets**: `RemoveMessage(id=REMOVE_ALL_MESSAGES)` clears transient
  history once it is summarized, preventing token bloat.
- **Operational Visibility**: Demonstrates `graph.get_state_history`, plus
  `checkpointer.get_tuple`, `list`, and `delete_thread` for observability.

=== COMPARISON WITH BASIC FILTERING ===
| agent_with_message_filter | Advanced Management (this file) |
|---------------------------|---------------------------------|
| Hand-written trimming only | LangMem-powered summarization node |
| No full-history reset      | Complete wipe using REMOVE_ALL_MESSAGES |
| Minimal checkpoint introspection | Full CRUD access to checkpoint metadata |

What You'll Learn
1. Integrate LangMem summarization into an agent while keeping raw history clean.
2. Reset message state once summaries exist using `REMOVE_ALL_MESSAGES`.
3. Inspect, list, and delete checkpoints to support production monitoring.

Lesson Flow
1. Build a graph that runs `SummarizationNode` before the LLM turn.
2. After responding, purge raw messages so only summaries persist.
3. Execute a demo conversation and exercise checkpoint inspection APIs.
"""

from __future__ import annotations

from typing import Iterable, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, RemoveMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from langmem.short_term import RunningSummary, SummarizationNode

from src.langgraph_learning.utils import (
    create_llm,
    maybe_enable_langsmith,
    pretty_print_messages,
    require_llm_provider_api_key,
    save_graph_image,
)


class SummaryState(MessagesState):
    """State schema expected by LangMem summarization."""

    context: dict[str, RunningSummary]


class LLMInputState(TypedDict):
    """Private state emitted by SummarizationNode before the call_model node."""

    summarized_messages: list[AnyMessage]
    context: dict[str, RunningSummary]


def build_advanced_graph(model: ChatOpenAI | None = None):
    """Compile a graph with LangMem summarization and reset controls."""
    llm = model or create_llm()
    summarizer_llm = llm.bind(max_tokens=256)

    summarization_node = SummarizationNode(
        token_counter=count_tokens_approximately,
        model=summarizer_llm,
        max_tokens=256,
        max_tokens_before_summary=256,
        max_summary_tokens=160,
    )

    def call_model(state: LLMInputState):
        response = llm.invoke(state["summarized_messages"])
        return {"messages": [response]}

    def cleanup_messages(state: SummaryState):
        if not state.get("context"):
            return {}
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}

    builder = StateGraph(SummaryState)
    builder.add_node("summarize", summarization_node)
    builder.add_node("assistant", call_model)
    builder.add_node("cleanup", cleanup_messages)
    builder.add_edge(START, "summarize")
    builder.add_edge("summarize", "assistant")
    builder.add_edge("assistant", "cleanup")
    builder.add_edge("cleanup", END)

    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)
    save_graph_image(
        graph, filename="artifacts/agent_with_advanced_memory_management.png"
    )
    return graph


def run_conversation(graph, prompts: Iterable[str], thread_id: str) -> None:
    """Stream a conversation and print responses."""
    config = {"configurable": {"thread_id": thread_id}}
    for text in prompts:
        events = graph.stream(
            {"messages": [HumanMessage(content=text)]},
            config,
            stream_mode="values",
        )
        for event in events:
            pretty_print_messages(
                event["messages"][-1:], header=f"Assistant reply to: {text}"
            )


def inspect_checkpoints(graph, thread_id: str) -> None:
    """Demonstrate checkpoint inspection and cleanup calls."""
    config = {"configurable": {"thread_id": thread_id}}
    history = list(graph.get_state_history(config))
    print("\n--- Checkpoint history ---")
    for snapshot in history:
        checkpoint_id = snapshot.config["configurable"]["checkpoint_id"]
        next_nodes = snapshot.next
        print(f"Checkpoint {checkpoint_id}, pending={next_nodes}")

    latest = graph.checkpointer.get_tuple(config)
    print("\n--- Latest checkpoint tuple ---")
    print(f"id={latest.checkpoint['id']}")
    print(f"writes={latest.metadata.get('writes')}")

    all_versions = list(graph.checkpointer.list(config))
    print("\n--- checkpointer.list output ---")
    for item in all_versions:
        print(
            f"id={item.checkpoint['id']} created_at={item.checkpoint['ts']} "
            f"channels={list(item.checkpoint['channel_values'].keys())}"
        )

    print("\n--- Deleting thread checkpoints ---")
    graph.checkpointer.delete_thread(thread_id)
    print("Thread deleted. Subsequent inspection would show no snapshots.")


def main() -> None:
    require_llm_provider_api_key()
    maybe_enable_langsmith()
    graph = build_advanced_graph()
    prompts = [
        "Hi, I'm Leslie and I run a robotics club for kids.",
        "Remember that we meet every Saturday morning.",
        "Summarize what you know about my club and ask a follow-up question.",
    ]
    thread_id = "advanced-memory-thread"
    run_conversation(graph, prompts, thread_id)
    inspect_checkpoints(graph, thread_id)


if __name__ == "__main__":
    main()
