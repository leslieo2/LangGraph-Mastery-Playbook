"""
Subgraph Memory Isolation: Preventing Cross-Agent Contamination

=== PROBLEM STATEMENT ===
Multi-agent workflows often share one parent graph. If every agent reuses the
same checkpointer, their message histories bleed together, leading to leaked
context and incorrect follow-ups.

=== CORE SOLUTION ===
This lesson compares two parent graphs: one where a subgraph inherits the
parent’s checkpointer, and another compiled with `checkpointer=True` so the
agent keeps its own conversation history.

=== KEY INNOVATION ===
- **Toggleable Isolation**: The same subgraph builder demonstrates inherited
  versus isolated memory by flipping a flag.
- **Configurable Agents**: `agent_id` travels through `RunnableConfig`,
  mimicking multi-agent routing in production systems.
- **State Inspection**: After each run, the lesson prints the recovered message
  history so learners can see whether cross-talk occurred.

=== COMPARISON WITH MONOLITHIC MEMORY ===
| Shared Checkpointer | Isolated Subgraph (this file) |
|---------------------|-------------------------------|
| All agents share one history | Each agent maintains its own transcript |
| Hard to audit per-agent runs | Namespaces separate checkpoints |
| Risk of leaking private info | Memory scoped to the responsible agent |

What You'll Learn
1. Compile subgraphs that either inherit parent memory or keep their own state.
2. Pass agent metadata through `RunnableConfig` to specialize subgraph behavior.
3. Inspect stored messages to verify when subgraph isolation is working.

Lesson Flow
1. Build an agent subgraph that echoes prior messages for the current agent.
2. Compile parent graphs with and without subgraph isolation.
3. Run two agents in sequence and compare the resulting transcripts.
"""

from __future__ import annotations

from typing import Iterable

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from src.langgraph_learning.utils import (
    create_llm,
    pretty_print_messages,
    require_llm_provider_api_key,
    save_graph_image,
)


def build_agent_subgraph(isolated: bool):
    """Compile a subgraph that replays prior turns for its assigned agent."""
    llm = create_llm()

    def analyst(state: MessagesState, config: RunnableConfig):
        agent_id = config["configurable"]["agent_id"]
        history = "\n".join(msg.content for msg in state["messages"]) or "No history yet."
        prompt = f"You are agent {agent_id}. Conversation history:\n{history}"
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"messages": [AIMessage(content=response.content)]}

    sub_builder = StateGraph(MessagesState)
    sub_builder.add_node("analyst", analyst)
    sub_builder.add_edge(START, "analyst")
    sub_builder.add_edge("analyst", END)

    compiled = (
        sub_builder.compile(checkpointer=True)
        if isolated
        else sub_builder.compile()
    )
    save_graph_image(
        compiled,
        filename=f"artifacts/subgraph_agent_isolated_{isolated}.png",
        xray=True,
    )
    return compiled


def build_parent_graph(isolated_subgraph: bool):
    """Create a parent graph that delegates to the agent subgraph."""
    agent = build_agent_subgraph(isolated_subgraph)

    def delegate(state: MessagesState):
        return {"messages": state["messages"]}

    parent_builder = StateGraph(MessagesState)
    parent_builder.add_node("delegate", delegate)
    parent_builder.add_node("agent", agent)
    parent_builder.add_edge(START, "delegate")
    parent_builder.add_edge("delegate", "agent")
    parent_builder.add_edge("agent", END)

    graph = parent_builder.compile(checkpointer=MemorySaver())
    save_graph_image(
        graph,
        filename=f"artifacts/parent_with_isolated_{isolated_subgraph}.png",
        xray=True,
    )
    return graph


def run_agents(graph, agent_sequence: Iterable[tuple[str, str]]) -> None:
    """Invoke the graph for multiple agents and print their outputs."""
    for agent_id, utterance in agent_sequence:
        config = {
            "configurable": {
                "thread_id": "shared-ticket",
                "agent_id": agent_id,
            }
        }
        events = graph.stream(
            {"messages": [HumanMessage(content=utterance)]},
            config,
            stream_mode="values",
        )
        for event in events:
            pretty_print_messages(
                event["messages"][-1:], header=f"Agent {agent_id} sees"
            )


def main() -> None:
    require_llm_provider_api_key()

    shared_graph = build_parent_graph(isolated_subgraph=False)
    isolated_graph = build_parent_graph(isolated_subgraph=True)

    sequence = [
        ("alpha", "Alpha here. Capture this requirement."),
        ("beta", "Beta joining. Show me only my notes."),
    ]

    print("\n=== Shared subgraph memory ===")
    run_agents(shared_graph, sequence)

    print("\n=== Isolated subgraph memory ===")
    run_agents(isolated_graph, sequence)


if __name__ == "__main__":
    main()

