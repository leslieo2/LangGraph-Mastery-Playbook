"""This LangGraph agent loops between an assistant LLM and arithmetic tools to execute multistep math requests with reactive routing.

What You'll Learn
1. Build a reactive agent loop that alternates between an assistant LLM node and tool execution.
2. Bind multiple arithmetic tools to a chat model and let LangGraph route tool calls automatically.
3. Persist graph diagrams and run a sample request end-to-end with pretty-printed messages.

Lesson Flow
1. Prepare reusable math tools and bind them to `ChatOpenAI`.
2. Define the assistant node, add the prebuilt `ToolNode`, and connect edges with `tools_condition`.
3. Compile the graph with a memory checkpointer, save a visualization, and invoke it with a multi-step prompt.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.langgraph_learning.utils import (
    create_llm,
    add,
    divide,
    multiply,
    pretty_print_messages,
    require_llm_provider_api_key,
    save_graph_image,
)


def build_agent_graph(model: str | None = None):
    """Compile a reactive graph that routes between an assistant node and tools."""
    tools = [add, multiply, divide]
    llm = create_llm(model=model)
    llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)
    sys_msg = SystemMessage(
        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
    )

    def assistant(state: MessagesState):
        return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("assistant", assistant)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "assistant")
    graph.add_conditional_edges("assistant", tools_condition)
    graph.add_edge("tools", "assistant")
    return graph.compile()


def run_demo(app) -> None:
    """Invoke the reactive agent on a sample arithmetic request."""
    messages = [
        HumanMessage(
            content="Add 3 and 4. Multiply the output by 2. Divide the output by 5."
        )
    ]
    result = app.invoke({"messages": messages})
    pretty_print_messages(result["messages"], header="Agent response")


def main() -> None:
    require_llm_provider_api_key()
    app = build_agent_graph()
    save_graph_image(app, filename="artifacts/agent_with_reactive_router.png", xray=True)
    run_demo(app)


if __name__ == "__main__":
    main()
