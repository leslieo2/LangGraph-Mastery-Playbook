"""This LangGraph streaming assistant alternates between live conversation and summarization while exposing raw chunks, value snapshots, and async events.

What You'll Learn
1. Enable streaming across LangGraph, including raw chunk updates and value snapshots.
2. Combine conversation summarization with checkpointing for long-lived streaming chats.
3. Monitor flows through async event streams.

Lesson Flow
1. Build a graph that alternates between conversation turns and summary pruning.
2. Exercise different streaming modes (`updates`, `values`, async events) with sample prompts.
3. Show how to resume runs, inspect tokens.
"""

from __future__ import annotations

import asyncio
from typing import Iterable
from uuid import uuid4

from langchain_core.messages import (
    AIMessageChunk,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    convert_to_messages,
)
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from src.langgraph_learning.utils import (
    create_llm,
    maybe_enable_langsmith,
    pretty_print_messages,
    require_llm_provider_api_key,
    save_graph_image,
)

try:  # Optional dependency for API streaming examples.
    from langgraph_sdk import get_client

    HAS_SDK = True
except ImportError:
    HAS_SDK = False


class State(MessagesState):
    summary: str


def build_streaming_graph(model: ChatOpenAI):
    def call_model(state: State, config: RunnableConfig | None = None):
        summary = state.get("summary", "")
        if summary:
            system_message = SystemMessage(
                content=f"Summary of conversation earlier: {summary}"
            )
            messages = [system_message, *state["messages"]]
        else:
            messages = state["messages"]

        response = model.invoke(messages, config=config)
        return {"messages": response}

    def summarize_conversation(state: State):
        summary = state.get("summary", "")

        if summary:
            summary_prompt = (
                f"This is summary of the conversation to date: {summary}\n\n"
                "Extend the summary by taking into account the new messages above:"
            )
        else:
            summary_prompt = "Create a summary of the conversation above:"

        messages = [*state["messages"], HumanMessage(content=summary_prompt)]
        response = model.invoke(messages)

        # After summarizing, drop older exchanges but keep the freshest context in place.
        delete_messages = [RemoveMessage(id=msg.id) for msg in state["messages"][:-2]]
        return {"summary": response.content, "messages": delete_messages}

    def should_continue(state: State) -> str:
        return "summarize_conversation" if len(state["messages"]) > 6 else END

    workflow = StateGraph(State)
    workflow.add_node("conversation", call_model)
    workflow.add_node("summarize_conversation", summarize_conversation)
    workflow.add_edge(START, "conversation")
    workflow.add_conditional_edges("conversation", should_continue)
    workflow.add_edge("summarize_conversation", END)

    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)

    save_graph_image(graph, filename="artifacts/streaming_chatbot.png")
    return graph


def _thread_config(prefix: str) -> dict:
    return {"configurable": {"thread_id": f"{prefix}-{uuid4().hex}"}}


def stream_updates_raw(graph) -> None:
    config = _thread_config("updates-raw")
    input_message = HumanMessage(content="hi! I'm Leslie")
    print("\n--- stream_mode='updates' (raw chunks) ---")
    for chunk in graph.stream(
        {"messages": [input_message]}, config, stream_mode="updates"
    ):
        print(chunk)


def stream_updates_messages(graph) -> None:
    config = _thread_config("updates-pretty")
    input_message = HumanMessage(content="hi! I'm Leslie")
    print("\n--- stream_mode='updates' (messages) ---")
    for chunk in graph.stream(
        {"messages": [input_message]}, config, stream_mode="updates"
    ):
        node_state = chunk.get("conversation")
        if node_state and (message := node_state.get("messages")):
            message.pretty_print()


def stream_values(graph) -> None:
    config = _thread_config("values")
    input_message = HumanMessage(content="hi! I'm Leslie")
    print("\n--- stream_mode='values' ---")
    for chunk in graph.stream(
        {"messages": [input_message]}, config, stream_mode="values"
    ):
        messages = chunk.get("messages", [])
        if messages:
            pretty_print_messages(messages, header="State snapshot")
        print("-" * 40)


async def stream_event_overview(graph) -> None:
    config = _thread_config("events-overview")
    input_message = HumanMessage(content="Tell me about the 49ers NFL team")
    print("\n--- astream_events overview ---")
    async for event in graph.astream_events(
        {"messages": [input_message]}, config, version="v2"
    ):
        node = event.get("metadata", {}).get("langgraph_node", "")
        name = event.get("name", "")
        print(f"Node: {node} | Event: {event['event']} | Name: {name}")


async def stream_chat_model_tokens(graph, node_to_stream: str = "conversation") -> None:
    config = _thread_config("events-tokens")
    input_message = HumanMessage(content="Tell me about the 49ers NFL team")
    print(f"\n--- Chat model tokens from node '{node_to_stream}' ---")
    async for event in graph.astream_events(
        {"messages": [input_message]}, config, version="v2"
    ):
        if (
            event["event"] == "on_chat_model_stream"
            and event.get("metadata", {}).get("langgraph_node") == node_to_stream
        ):
            data = event.get("data", {})
            chunk = data.get("chunk")
            if isinstance(chunk, AIMessageChunk):
                print(chunk.content, end="|")
            else:
                print(chunk, end="|")
    print()


async def stream_chat_model_chunks(graph, node_to_stream: str = "conversation") -> None:
    config = _thread_config("events-chunks")
    input_message = HumanMessage(content="Tell me about the 49ers NFL team")
    print(f"\n--- Chat model chunk objects from node '{node_to_stream}' ---")
    async for event in graph.astream_events(
        {"messages": [input_message]}, config, version="v2"
    ):
        if (
            event["event"] == "on_chat_model_stream"
            and event.get("metadata", {}).get("langgraph_node") == node_to_stream
        ):
            print(event.get("data"))


def main() -> None:
    require_llm_provider_api_key()
    maybe_enable_langsmith()
    model = create_llm()

    graph = build_streaming_graph(model)

    stream_updates_raw(graph)
    stream_updates_messages(graph)
    stream_values(graph)

    asyncio.run(stream_event_overview(graph))
    asyncio.run(stream_chat_model_tokens(graph, "conversation"))
    asyncio.run(stream_chat_model_chunks(graph))


if __name__ == "__main__":
    main()
