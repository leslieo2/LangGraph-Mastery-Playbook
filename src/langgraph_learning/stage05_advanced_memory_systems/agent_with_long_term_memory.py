"""
Long-Term Memory Agent: Bullet-Point Reflections for Personalized Replies

=== PROBLEM STATEMENT ===
Conversational agents quickly forget user details when each turn is treated in isolation.
Without persistent memory, they repeat questions, miss preferences, and lose continuity
across sessions—especially when the same user returns in a new thread.

=== CORE SOLUTION ===
This lesson builds a LangGraph that reads long-term memory before responding and writes
structured bullet summaries after each turn. An `InMemoryStore` keeps user-level notes,
while `MemorySaver` maintains per-thread state for reliable conversation playback.

=== KEY INNOVATION ===
- **Cross-Thread Personalization**: Memory is keyed by `user_id`, so knowledge survives
  across conversations.
- **Reflection Loop**: A dedicated write node distills the latest chat into clear,
  bulleted facts.
- **Config-Driven Access**: `RunnableConfig` provides both `thread_id` and `user_id`,
  demonstrating how LangGraph configuration steers storage and retrieval.

=== COMPARISON WITH STRUCTURED PROFILES ===
| Structured Profiles (TrustCall) | Long-Term Bullet Memory (this file) |
|---------------------------------|-------------------------------------|
| JSON schemas with field-level updates | Free-form bullet list stored as text |
| TrustCall manages incremental patches | Model regenerates entire summary each turn |
| Rich structure for downstream automation | Lightweight notes for quick personalization |

What You'll Learn
1. Personalize responses by reading and writing user memory within a LangGraph workflow.
2. Reflect on recent conversation turns to synthesize structured memory snippets.
3. Demonstrate how thread-specific configuration drives both retrieval and storage.

Lesson Flow
1. Define system prompts for reading memory and for reflecting on new information.
2. Build a two-node graph (`call_model` → `write_memory`) backed by `InMemoryStore` + `MemorySaver`.
3. Compile the graph, render a diagram, and execute consecutive turns to observe memory updates.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.append(project_root)

from src.langgraph_learning.stage05_advanced_memory_systems.configuration import (
    MemoryConfiguration,
)
from src.langgraph_learning.utils import (
    create_llm,
    llm_from_config,
    maybe_enable_langsmith,
    pretty_print_messages,
    require_llm_provider_api_key,
    save_graph_image,
)

MODEL_SYSTEM_MESSAGE = """You are a helpful assistant with memory that provides information about the user.
If you have memory for this user, use it to personalize your responses.
Here is the memory (it may be empty): {memory}"""

CREATE_MEMORY_INSTRUCTION = """You are collecting information about the user to personalize your responses.

Current user information:
{memory}

Instructions:
1. Review the chat history below carefully.
2. Identify new information about the user (name, location, preferences, hobbies, experiences, goals).
3. Merge any new information with existing memory.
4. Format the memory as a clear, bulleted list.
5. If new information conflicts with existing memory, keep the most recent version.

Only include factual information stated by the user. Do not make assumptions.
Update the user information based on the chat history below:"""


def _assemble_memory_graph(
    llm: ChatOpenAI,
    *,
    store: BaseStore | None = None,
    checkpointer: MemorySaver | None = None,
):
    """Return compiled long-term memory graph using supplied dependencies."""
    store = store or InMemoryStore()
    checkpointer = checkpointer or MemorySaver()

    def _config(config: RunnableConfig) -> MemoryConfiguration:
        return MemoryConfiguration.from_runnable_config(config)

    def call_model(state: MessagesState, config: RunnableConfig, store: BaseStore):
        # Only pull user-level settings; thread_id stays in the original config for MemorySaver.
        cfg = _config(config)
        namespace = ("memory", cfg.user_id)
        key = "user_memory"
        existing = store.get(namespace, key)
        memory_text = (
            existing.value.get("memory") if existing else "No existing memory found."
        )
        system_msg = MODEL_SYSTEM_MESSAGE.format(memory=memory_text)
        response = llm.invoke([SystemMessage(content=system_msg), *state["messages"]])
        return {"messages": response}

    def write_memory(state: MessagesState, config: RunnableConfig, store: BaseStore):
        # Same idea here—thread_id remains on config while we use user_id for storage.
        cfg = _config(config)
        namespace = ("memory", cfg.user_id)
        existing = store.get(namespace, "user_memory")
        memory_text = (
            existing.value.get("memory") if existing else "No existing memory found."
        )
        system_msg = CREATE_MEMORY_INSTRUCTION.format(memory=memory_text)
        reflection = llm.invoke([SystemMessage(content=system_msg), *state["messages"]])
        store.put(namespace, "user_memory", {"memory": reflection.content})

    builder = StateGraph(MessagesState, config_schema=MemoryConfiguration)
    builder.add_node("call_model", call_model)
    builder.add_node("write_memory", write_memory)
    builder.add_edge(START, "call_model")
    builder.add_edge("call_model", "write_memory")
    builder.add_edge("write_memory", END)

    return builder.compile(store=store, checkpointer=checkpointer)


def build_memory_graph(
    model: ChatOpenAI | None = None,
    *,
    store: BaseStore | None = None,
    checkpointer: MemorySaver | None = None,
):
    """Compile a graph that reflects on conversations and stores user memory."""
    llm = model or create_llm()
    graph = _assemble_memory_graph(
        llm,
        store=store,
        checkpointer=checkpointer,
    )
    save_graph_image(
        graph, filename="artifacts/agent_with_long_term_memory.png", xray=True
    )
    return graph


def demo_conversation(graph) -> None:
    """Run a short conversation that populates and reads from memory."""
    config = {"configurable": {"thread_id": "thread-1", "user_id": "leslie"}}
    turns = [
        "Hi, my name is Leslie.",
        "I like to bike around San Francisco.",
        "Great! Recommend a weekend route for me.",
    ]
    for text in turns:
        messages = [HumanMessage(content=text)]
        for chunk in graph.stream({"messages": messages}, config, stream_mode="values"):
            pretty_print_messages(
                chunk["messages"][-1:], header=f"Assistant reply to: {text}"
            )


def main() -> None:
    require_llm_provider_api_key()
    maybe_enable_langsmith()
    graph = build_memory_graph()
    demo_conversation(graph)


if __name__ == "__main__":
    main()


def studio_graph(config: RunnableConfig | None = None):
    """Studio entry point for the long-term memory agent."""
    llm, _ = llm_from_config(config)
    return _assemble_memory_graph(llm)
