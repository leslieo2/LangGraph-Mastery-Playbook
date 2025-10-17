"""This LangGraph mini-agent binds a chat model to a multiply tool and highlights how message history flows through a single-node tool-calling workflow.

What You'll Learn
1. Review how LangGraph reuses existing chat messages and how `add_messages` manages history.
2. See how to bind a LangChain chat model to a simple Python tool for entry-level tool calling.
3. Compile a single-node `MessagesState` graph, render its structure, and run an end-to-end LLM flow.

Lesson Flow
1. Assemble a sample conversation and inspect the message structure with `pretty_print_messages`.
2. Bind a multiply tool to the model, trigger the tool, and inspect the response.
3. Demonstrate how `add_messages` appends or overwrites existing history.
4. Build a minimal LangGraph app, save a diagram, and send one message through the graph.
"""

from __future__ import annotations

from typing import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph, add_messages

from src.langgraph_learning.utils import (
    create_llm,
    multiply,
    pretty_print_messages,
    require_llm_provider_api_key,
    save_graph_image,
)

Messages = Sequence[BaseMessage]


def sample_transcript() -> Messages:
    """Return a fixed conversation snippet used across the demos."""
    return (
        AIMessage(
            content="So you said you were researching ocean mammals?", name="Model"
        ),
        HumanMessage(content="Yes, that's right.", name="Leslie"),
        AIMessage(content="Great, what would you like to learn about.", name="Model"),
        HumanMessage(content="Where can I see Orcas in the US?", name="Leslie"),
    )


def inspect_messages(model: str | None = None) -> None:
    """Show how to invoke an LLM on a list of messages."""
    llm = create_llm(model=model)
    history = list(sample_transcript())
    pretty_print_messages(history, header="Input messages")

    response = llm.invoke(history)
    print("\nModel reply:", response.content)


def inspect_tool_binding(model: str | None = None) -> None:
    """Demonstrate how to bind simple Python tools to a chat model."""
    llm = create_llm(model=model)
    llm_with_tools = llm.bind_tools([multiply])
    response = llm_with_tools.invoke(
        [HumanMessage(content="What is 2 multiplied by 3?")]
    )
    pretty_print_messages([response], header="Tool call response")


def inspect_add_messages() -> None:
    """Illustrate append/overwrite semantics of `add_messages`."""
    initial_messages = list(sample_transcript())[:2]
    new_message = AIMessage(
        content="Sure. What specifically are you interested in?", name="Model"
    )
    result = add_messages(initial_messages, new_message)
    print("\nCombined messages:", result)


def build_tool_calling_app(model: str | None = None):
    """Create a compiled graph that routes messages through the LLM."""
    llm = create_llm(model=model)
    llm_with_tools = llm.bind_tools([multiply])

    def llm_node(state: MessagesState):
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("tool_calling_llm", llm_node)
    graph.add_edge(START, "tool_calling_llm")
    graph.add_edge("tool_calling_llm", END)
    return graph.compile()


def run_app_demo(app) -> None:
    """Send a sample message through the app and display the response."""
    result = app.invoke({"messages": [HumanMessage(content="hello", name="Leslie")]})
    pretty_print_messages(result["messages"], header="Graph output")


def main() -> None:
    require_llm_provider_api_key()
    inspect_messages()
    inspect_tool_binding()
    inspect_add_messages()

    app = build_tool_calling_app()
    save_graph_image(app, filename="artifacts/agent_with_tool_call.png")
    run_app_demo(app)


if __name__ == "__main__":
    main()
