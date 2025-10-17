"""This LangGraph chatbot alternates between answering the user and rolling summaries so long conversations stay within token limits.

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


class State(MessagesState):
    summary: str


def build_chatbot_graph(model: ChatOpenAI):
    """Create a chatbot that summarizes older context into rolling memory."""
    memory = MemorySaver()

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

    graph = workflow.compile(checkpointer=memory)
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
