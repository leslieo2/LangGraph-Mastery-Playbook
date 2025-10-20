"""
Short-Term Memory: Checkpointing Conversational State

=== PROBLEM STATEMENT ===
Agents without memory forget previous answers, forcing users to restate context. Within a
single session, you need a lightweight way to carry over conversation history.

Memory Type: Short-term (conversation-level) memory
- Scope: Single-thread conversation context
- Storage: MemorySaver checkpointer preserves full message history
- Purpose: Maintain continuity within a session that shares the same `thread_id`

=== CORE SOLUTION ===
This lesson adds `MemorySaver` checkpoints to a reactive arithmetic agent so each call
reuses prior messages. Follow-up prompts can reference earlier results without redoing all
work.

=== KEY INNOVATION ===
- **Checkpoint Integration**: Compile the graph with `MemorySaver`.
- **Thread-Aware Invocations**: Use `thread_id` to load the right history.
- **Demonstrated Continuity**: Show that step two references step one automatically.

=== COMPARISON WITH STATELESS AGENTS ===
| Stateless Flow | Short-Term Memory (this file) |
|----------------|-------------------------------|
| Each call starts fresh | Checkpoints replay past exchanges |
| Users must repeat context | Follow-up prompts reuse stored messages |
| No notion of threads | `thread_id` ties invocations together |

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
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.langgraph_learning.utils import (
    add,
    create_llm,
    llm_from_config,
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


def _assemble_short_term_memory_graph(llm, memory: MemorySaver):
    """Return a compiled short-term memory graph using provided dependencies."""
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

    return graph.compile(checkpointer=memory)


def build_agent_graph(*, model: str | None = None):
    """Compile a reactive agent that remembers prior exchanges."""
    llm = create_llm(model=model)
    memory = MemorySaver()
    return _assemble_short_term_memory_graph(llm, memory)


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
    save_graph_image(
        app, filename="artifacts/agent_with_short_term_memory.png", xray=True
    )
    run_demo(app)


if __name__ == "__main__":
    main()


def studio_graph(config: RunnableConfig | None = None):
    """Studio entry point for the short-term memory agent."""
    llm, _ = llm_from_config(config)
    memory = MemorySaver()
    return _assemble_short_term_memory_graph(llm, memory)
