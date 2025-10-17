"""This LangGraph arithmetic agent keeps checkpoints between tool calls so it can reuse previous answers when the user follows up later.

What You'll Learn
1. Extend a reactive agent with checkpoints so it can remember prior turns.
2. Configure a `MemorySaver` checkpointer to maintain conversation state across invocations.
3. Demonstrate persistence by running sequential prompts that reference earlier results.

Lesson Flow
1. Reuse arithmetic tools and bind them to a chat model with tool calling disabled in parallel.
2. Build the assistant ↔ tool loop, compile with `MemorySaver`, and generate a graph image.
3. Invoke the agent twice under the same thread to confirm it carries forward intermediate outputs.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
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
    """Compile a reactive agent that remembers prior exchanges."""
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

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


def run_demo(app) -> None:
    """Demonstrate how the agent remembers context across turns."""
    config = {"configurable": {"thread_id": "memory-demo"}}

    first_turn = [HumanMessage(content="Add 3 and 4.")]
    response = app.invoke({"messages": first_turn}, config)
    pretty_print_messages(response["messages"], header="First turn")

    follow_up = [HumanMessage(content="Multiply the result by 2.")]
    response = app.invoke({"messages": follow_up}, config)
    pretty_print_messages(response["messages"], header="Second turn (with memory)")


def main() -> None:
    require_llm_provider_api_key()
    app = build_agent_graph()
    save_graph_image(app, filename="artifacts/agent_with_memory.png", xray=True)
    run_demo(app)


if __name__ == "__main__":
    main()
