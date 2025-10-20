"""
Reactive Tool Loop: Multi-Step Arithmetic Agent

=== PROBLEM STATEMENT ===
Chained tool calls require the agent to alternate between reasoning and execution. Without
an explicit loop, it’s hard to see how LangGraph lets LLMs plan, call tools, and iterate.

=== CORE SOLUTION ===
This lesson constructs a reactive agent that cycles between an assistant node and a
`ToolNode`, enabling the model to call multiple arithmetic tools in one conversation.

=== KEY INNOVATION ===
- **Assistant ↔ Tool Loop**: Connects nodes so tool outputs feed back into the model.
- **Parallel Control**: Disables parallel tool calls to keep sequencing deterministic.
- **Graph Visualization**: Saves an xray diagram to reinforce the reactive pattern.

=== COMPARISON WITH STATIC TOOL ROUTING ===
| Single Tool Call (one shot) | Reactive Loop (this file) |
|-----------------------------|---------------------------|
| One tool invocation per request | Multiple tool calls orchestrated by the model |
| No iterative reasoning | Model plans, executes, and updates state repeatedly |
| Simpler graphs | Closer to production agent behavior |

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
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.langgraph_learning.utils import (
    create_llm,
    llm_from_config,
    add,
    divide,
    multiply,
    pretty_print_messages,
    require_llm_provider_api_key,
    save_graph_image,
)


TOOLS = [add, multiply, divide]
SYSTEM_MESSAGE = SystemMessage(
    content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
)


def _assemble_reactive_tool_graph(llm):
    """Return compiled reactive agent graph using provided LLM."""
    llm_with_tools = llm.bind_tools(TOOLS, parallel_tool_calls=False)

    def assistant(state: MessagesState):
        return {
            "messages": [llm_with_tools.invoke([SYSTEM_MESSAGE] + state["messages"])]
        }

    graph = StateGraph(MessagesState)
    graph.add_node("assistant", assistant)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "assistant")
    graph.add_conditional_edges("assistant", tools_condition)
    graph.add_edge("tools", "assistant")
    return graph.compile()


def build_agent_graph(*, model: str | None = None):
    """Compile a reactive graph that routes between an assistant node and tools."""
    llm = create_llm(model=model)
    return _assemble_reactive_tool_graph(llm)


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
    save_graph_image(
        app, filename="artifacts/agent_with_reactive_router.png", xray=True
    )
    run_demo(app)


if __name__ == "__main__":
    main()


def studio_graph(config: RunnableConfig | None = None):
    """Studio entry point for the reactive router graph."""
    llm, _ = llm_from_config(config)
    return _assemble_reactive_tool_graph(llm)
