"""
Tool Routing: Detect, Execute, and Return Results

=== PROBLEM STATEMENT ===
Simple tool demos often hide how LangGraph detects model-issued tool calls and routes
them to actual executors. Learners need a clear example that splits tool detection from
execution.

=== CORE SOLUTION ===
This lesson binds a multiply tool to `ChatOpenAI`, uses `tools_condition` to branch into a
`ToolNode`, and returns the updated conversation, illustrating the full detection →
execution → resume loop.

=== KEY INNOVATION ===
- **Tool Detection**: Showcases conditional routing via `tools_condition`.
- **Execution Node**: Leverages `ToolNode` to handle the actual multiply call.
- **Augmented Transcript**: Returns the conversation with tool results included.

=== COMPARISON WITH SINGLE-NODE TOOL CALL ===
| Single Node Tool Call | Tool Router (this file) |
|-----------------------|-------------------------|
| Tool invocation happens inside the assistant node | Dedicated ToolNode executes selected tool |
| No routing logic | `tools_condition` handles dynamic branching |
| Simpler but less flexible | Closer to production agent architecture |

What You'll Learn
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
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.langgraph_learning.utils import (
    create_llm,
    multiply,
    pretty_print_messages,
    require_llm_provider_api_key,
    save_graph_image,
)


def build_tool_calling_graph(model: str | None = None):
    """Create a compiled StateGraph configured for tool routing."""
    llm = create_llm(model=model)
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
    require_llm_provider_api_key()
    graph = build_tool_calling_graph()
    save_graph_image(graph, filename="artifacts/agent_with_tool_router.png")
    run_demo(graph)


if __name__ == "__main__":
    main()
