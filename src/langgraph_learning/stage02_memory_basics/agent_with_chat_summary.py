"""
Chat Summaries: Balancing Context Length and Memory

=== PROBLEM STATEMENT ===
Conversations grow beyond token budgets. Without summarization, agents either truncate
context blindly or blow past limits.

=== CORE SOLUTION ===
This lesson alternates between a conversation node and a summarizer node. Older exchanges
are distilled into a rolling summary using `MemorySaver`, keeping prompts concise while
retaining key facts.

=== KEY INNOVATION ===
- **Summary Injection**: Prepend the latest summary before invoking the model.
- **Reducer Cleanup**: Use `RemoveMessage` instructions to drop stale turns.
- **Memory Persistence**: Store summaries per thread via `MemorySaver`.

=== COMPARISON WITH RAW HISTORY ===
| Raw Transcript | Summarized Flow (this file) |
|----------------|-----------------------------|
| Tokens grow unchecked | Summaries keep prompts under budget |
| Hard to reason about pruning | Explicit summarizer node handles cleanup |
| No cross-turn reinforcement | Stored summary survives across invocations |

What You'll Learn
1. Maintain long-running conversations by summarizing older context into compact memory.
2. Apply LangGraph reducers to trim message history while preserving salient details.
3. Use `MemorySaver` to retain summaries across threads and inspect the stored state.

Lesson Flow
1. Build a conversation node that optionally prepends the current summary before invoking the model.
2. Add a summarization node that condenses recent exchanges and prunes stale messages.
3. Compile the graph with memory, run multiple prompts, and print both replies and the evolving summary.
"""

from __future__ import annotations

from typing import Iterable, Literal

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from src.langgraph_learning.utils import (
    create_llm,
    llm_from_config,
    maybe_enable_langsmith,
    pretty_print_messages,
    require_llm_provider_api_key,
    save_graph_image,
)


class State(MessagesState):
    summary: str


def _assemble_chat_summary_graph(model: ChatOpenAI, memory: MemorySaver):
    """Return compiled chatbot graph with supplied model and memory."""

    def conversation(state: State):
        summary = state.get("summary", "")
        if summary:
            system_message = SystemMessage(
                content=f"Summary of conversation earlier: {summary}"
            )
            messages: list[AnyMessage] = [system_message, *state["messages"]]
        else:
            messages = state["messages"]

        response = model.invoke(messages)
        return {"messages": response}

    def summarize_conversation(state: State):
        summary = state.get("summary", "")
        summary_prompt = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
            if summary
            else "Create a summary of the conversation above:"
        )

        messages = [*state["messages"], HumanMessage(content=summary_prompt)]
        response = model.invoke(messages)
        delete_messages = [RemoveMessage(id=msg.id) for msg in state["messages"][:-2]]
        return {"summary": response.content, "messages": delete_messages}

    def should_summary(state: State) -> Literal["summarize_conversation", END]:
        return "summarize_conversation" if len(state["messages"]) > 6 else END

    workflow = StateGraph(State)
    workflow.add_node("conversation", conversation)
    workflow.add_node("summarize_conversation", summarize_conversation)
    workflow.add_edge(START, "conversation")
    workflow.add_conditional_edges("conversation", should_summary)
    workflow.add_edge("summarize_conversation", END)

    return workflow.compile(checkpointer=memory)


def build_chatbot_graph(model: ChatOpenAI):
    """Create a chatbot that summarizes older context into rolling memory."""
    graph = _assemble_chat_summary_graph(model, MemorySaver())
    save_graph_image(graph, filename="artifacts/agent_with_chat_summary.png")
    return graph


def run_conversation(graph, prompts: Iterable[str], thread_id: str) -> None:
    config = {"configurable": {"thread_id": thread_id}}

    for text in prompts:
        input_message = HumanMessage(content=text)
        output = graph.invoke({"messages": [input_message]}, config)
        pretty_print_messages(
            output["messages"][-1:], header=f"Assistant reply to: {text}"
        )

    summary = graph.get_state(config).values.get("summary", "")
    if summary:
        print("\nCurrent summary:\n", summary)
    else:
        print("\nNo summary stored yet. Fewer than threshold messages.")


def main() -> None:
    require_llm_provider_api_key()
    maybe_enable_langsmith()
    model = create_llm()

    graph = build_chatbot_graph(model)

    initial_prompts = [
        "hi! I'm Leslie",
        "what's my name?",
        "i like the 49ers!",
    ]
    run_conversation(graph, initial_prompts, thread_id="1")

    follow_up_prompts = [
        "i like Nick Bosa, isn't he the highest paid defensive player?",
    ]
    run_conversation(graph, follow_up_prompts, thread_id="1")


if __name__ == "__main__":
    main()


def studio_graph(config: RunnableConfig | None = None):
    """Studio entry point for the chat summary agent."""
    llm, _ = llm_from_config(config)
    return _assemble_chat_summary_graph(llm, MemorySaver())
