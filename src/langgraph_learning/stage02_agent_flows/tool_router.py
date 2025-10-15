"""What You'll Learn
1. Configure a LangGraph workflow that detects tool calls and routes control to a tool node.
2. Bind a simple multiply tool to `ChatOpenAI` and observe automatic tool invocation.
3. Save graph visuals and run a demo request to inspect the routed response.

Lesson Flow
1. Bind the multiply tool to the chat model and define the tool-calling node function.
2. Add the `ToolNode`, configure conditional edges via `tools_condition`, and compile the graph.
3. Render the graph to PNG, execute a sample query, and pretty-print the assistant reply.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.langgraph_learning.utils import (
    multiply,
    pretty_print_messages,
    require_env,
    save_graph_image,
)


def build_tool_calling_graph(model: str = "gpt-5-nano"):
    """Create a compiled StateGraph configured for tool routing."""
    llm = ChatOpenAI(model=model)
    llm_with_tools = llm.bind_tools([multiply])

    def tool_calling_llm(state: MessagesState):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("tool_calling_llm", tool_calling_llm)
    graph.add_node("tools", ToolNode([multiply]))
    graph.add_edge(START, "tool_calling_llm")
    graph.add_conditional_edges("tool_calling_llm", tools_condition)
    graph.add_edge("tools", END)
    return graph.compile()


def run_demo(graph) -> None:
    """Run a small demo conversation through the compiled graph."""
    messages = [HumanMessage(content="Hello, what is 2 multiplied by 2?")]
    result = graph.invoke({"messages": messages})
    pretty_print_messages(result["messages"], header="Router output")


def main() -> None:
    require_env("OPENAI_API_KEY")
    graph = build_tool_calling_graph()
    save_graph_image(graph, filename="artifacts/tool_router.png")
    run_demo(graph)


if __name__ == "__main__":
    main()
