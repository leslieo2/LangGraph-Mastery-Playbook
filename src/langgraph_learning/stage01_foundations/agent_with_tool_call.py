"""
Single-Node Tool Call: MessagesState Warm-Up

A simple demonstration of tool calling with LangGraph.

Key Concepts:
- bind_tools() makes LLMs aware of available tools (does NOT execute them)
- Use MessagesState to manage conversation history
- Build minimal single-node graphs for tool calling

What You'll Learn:
1. How bind_tools() works - it only informs the model about tools
2. How MessagesState manages conversation flow
3. How to build and run simple tool-calling graphs

Important Note:
bind_tools() tells the model "these tools exist" but the model still needs to:
- Decide when to use tools
- Provide tool arguments
- Wait for actual tool execution (requires additional logic)
"""

from __future__ import annotations

from typing import Sequence

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph

from src.langgraph_learning.utils import (
    create_llm,
    llm_from_config,
    multiply,
    pretty_print_messages,
    require_llm_provider_api_key,
    save_graph_image,
)

Messages = Sequence[BaseMessage]


def inspect_tool_binding() -> None:
    """Demonstrate how to bind simple Python tools to a chat model.

    Important: bind_tools() only makes the model aware of available tools.
    The model can suggest tool usage, but doesn't automatically execute them.
    """
    llm = create_llm()
    # bind_tools() tells the model about available tools, but doesn't execute them
    llm_with_tools = llm.bind_tools([multiply])
    response = llm_with_tools.invoke(
        [HumanMessage(content="What is 223123123 multiplied by 3241241?")]
    )
    pretty_print_messages([response], header="Tool call response")
    print(
        "\nNote: The model knows about the multiply tool but doesn't execute it automatically."
    )


def build_tool_calling_app(*, model: str | None = None):
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


def studio_graph(config: RunnableConfig | None = None):
    """Studio entry point that reuses optional config overrides."""
    llm, _ = llm_from_config(config)
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
    """Run the tool calling demonstration."""
    require_llm_provider_api_key()

    # Show tool binding in action
    inspect_tool_binding()

    # Build and run the graph
    app = build_tool_calling_app()
    save_graph_image(app, filename="artifacts/agent_with_tool_call.png")
    run_app_demo(app)


if __name__ == "__main__":
    main()
