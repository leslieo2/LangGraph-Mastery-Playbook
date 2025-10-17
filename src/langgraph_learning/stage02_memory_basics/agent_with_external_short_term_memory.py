"""This LangGraph chatbot writes conversations and rolling summaries to SQLite so each session can pick up where the last one ended.

What You'll Learn
1. Persist LangGraph conversation state to SQLite for durable, cross-session memory.
2. Combine summaries with long-term storage so the assistant can recall prior threads.
3. Exercise the workflow with multiple conversation rounds and inspect saved records.

Lesson Flow
1. Prepare the schema-aware state graph with conversation and summarization nodes.
2. Construct a SQLite-backed `SqliteSaver`, compile the graph, and store the visualization.
3. Run scripted interactions under a shared `user_id`, then query the underlying store for updates.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, Literal

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph

from src.langgraph_learning.utils import (
    create_llm,
    maybe_enable_langsmith,
    pretty_print_messages,
    require_llm_provider_api_key,
    save_graph_image,
)


class State(MessagesState):
    summary: str


def build_chatbot_graph(model: ChatOpenAI, db_path: Path) -> StateGraph:
    """Create a chatbot that persists conversation state to SQLite."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)

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

    memory = SqliteSaver(conn)
    graph = workflow.compile(checkpointer=memory)
    save_graph_image(
        graph, filename="artifacts/agent_with_external_short_term_memory.png"
    )
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
    db_path = Path("artifacts/state_db/example.db")

    graph = build_chatbot_graph(model, db_path)

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
