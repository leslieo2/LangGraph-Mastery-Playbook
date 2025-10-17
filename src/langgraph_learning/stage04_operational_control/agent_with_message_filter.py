"""This LangGraph chat workflow contrasts reducer pruning, invocation-time filtering, and token trimming to control how much history the model sees.

What You'll Learn
1. Compare multiple strategies for managing conversation history in LangGraph.
2. Apply reducer-based filtering, selective invocation windows, and token-aware trimming.
3. Print message transcripts at each stage to understand what the model actually sees.

Lesson Flow
1. Run a baseline chat node that echoes the entire message list for reference.
2. Introduce a reducer node that deletes older messages before the model call.
3. Demonstrate invocation-time filtering and `trim_messages`, printing the resulting outputs.
"""

from __future__ import annotations

from typing import Iterable

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    trim_messages,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph

from src.langgraph_learning.utils import (
    create_llm,
    maybe_enable_langsmith,
    pretty_print_messages,
    require_llm_provider_api_key,
    save_graph_image,
)


def demonstrate_basic_messages(llm: ChatOpenAI):
    messages = [
        AIMessage("So you said you were researching ocean mammals?", name="Bot"),
        HumanMessage(
            "Yes, I know about whales. But what others should I learn about?",
            name="Leslie",
        ),
    ]
    pretty_print_messages(messages, header="Starting messages")

    def chat_model_node(state: MessagesState):
        return {"messages": llm.invoke(state["messages"])}

    builder = StateGraph(MessagesState)
    builder.add_node("chat_model", chat_model_node)
    builder.add_edge(START, "chat_model")
    builder.add_edge("chat_model", END)
    graph = builder.compile()
    output = graph.invoke({"messages": messages})
    pretty_print_messages(output["messages"], header="Model response")
    return messages, output


def demonstrate_filter_with_reducer(
    llm: ChatOpenAI,
) -> tuple[list[AIMessage | HumanMessage], dict]:
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

    def filter_messages(state: MessagesState):
        # Remove everything except the most recent two messages; this permanently
        # shrinks the conversation state, so later nodes never see the discarded
        # history.
        delete_messages = [RemoveMessage(id=msg.id) for msg in state["messages"][:-2]]
        return {"messages": delete_messages}

    def chat_model_node(state: MessagesState):
        return {"messages": [llm.invoke(state["messages"])]}

    builder = StateGraph(MessagesState)
    builder.add_node("filter", filter_messages)
    builder.add_node("chat_model", chat_model_node)
    builder.add_edge(START, "filter")
    builder.add_edge("filter", "chat_model")
    builder.add_edge("chat_model", END)
    graph = builder.compile()

    save_graph_image(graph, filename="artifacts/agent_with_message_reduce_filter.png")
    output = graph.invoke({"messages": messages})
    pretty_print_messages(output["messages"], header="After reducer filtering")
    return messages, output


def demonstrate_message_filtering(
    llm: ChatOpenAI,
    base_messages: Iterable[AIMessage | HumanMessage],
    previous_output: dict,
) -> tuple[list[AIMessage | HumanMessage], dict]:
    messages = list(base_messages)
    messages.append(previous_output["messages"][-1])
    messages.append(HumanMessage("Tell me more about Narwhals!", name="Leslie"))

    pretty_print_messages(messages, header="Conversation before filtering")

    def chat_model_node(state: MessagesState):
        # Only the final message is sent to the model, but the conversation state
        # itself remains untouched—useful when you want lightweight prompts without
        # losing history for logging or future logic.
        return {"messages": [llm.invoke(state["messages"][-1:])]}

    builder = StateGraph(MessagesState)
    builder.add_node("chat_model", chat_model_node)
    builder.add_edge(START, "chat_model")
    builder.add_edge("chat_model", END)
    graph = builder.compile()

    output = graph.invoke({"messages": messages})
    pretty_print_messages(output["messages"], header="Filtered invocation result")
    return messages, output


def demonstrate_trim_messages(
    llm: ChatOpenAI,
    base_messages: Iterable[AIMessage | HumanMessage],
    previous_output: dict,
) -> dict:
    messages = list(base_messages)
    messages.append(previous_output["messages"][-1])
    messages.append(HumanMessage("Tell me where Orcas live!", name="Leslie"))

    pretty_print_messages(messages, header="Conversation before trimming")

    example_trim = trim_messages(
        messages,
        max_tokens=100,
        strategy="last",
        token_counter=llm,  # LangChain helper that trims by token count instead of message count.
        allow_partial=False,
    )
    pretty_print_messages(example_trim, header="Preview of trimmed messages")

    def chat_model_node(state: MessagesState):
        # State keeps the full history; `trim_messages` enforces a 100-token window
        # (with allow_partial=False preserving message boundaries) so the model sees the
        # freshest context without exploding token usage.
        trimmed = trim_messages(
            state["messages"],
            max_tokens=100,
            strategy="last",
            token_counter=llm,
            allow_partial=False,
        )
        return {"messages": [llm.invoke(trimmed)]}

    builder = StateGraph(MessagesState)
    builder.add_node("chat_model", chat_model_node)
    builder.add_edge(START, "chat_model")
    builder.add_edge("chat_model", END)
    graph = builder.compile()

    save_graph_image(graph, filename="artifacts/agent_with_message_trim.png")
    output = graph.invoke({"messages": messages})
    pretty_print_messages(output["messages"], header="Trimming invocation result")
    return output


def main() -> None:
    require_llm_provider_api_key()
    maybe_enable_langsmith()
    llm = create_llm()

    _, basic_output = demonstrate_basic_messages(llm)
    reducer_messages, reducer_output = demonstrate_filter_with_reducer(llm)
    filtered_messages, filtered_output = demonstrate_message_filtering(
        llm, reducer_messages, reducer_output
    )
    demonstrate_trim_messages(llm, filtered_messages, filtered_output)


if __name__ == "__main__":
    main()
