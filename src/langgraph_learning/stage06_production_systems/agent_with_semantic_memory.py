"""
Semantic Memory: Embedding-Driven Retrieval inside LangGraph

=== PROBLEM STATEMENT ===
Keyword lookups miss paraphrases, and naive replay of every message overflows
token budgets. Production agents need semantic recall so user facts remain
discoverable even when phrased differently.

=== CORE SOLUTION ===
This lesson builds an `InMemoryStore` with an embedding index, writes memories in
response to “remember” requests, and retrieves semantically similar entries
before calling the model.

=== KEY INNOVATION ===
- **Vector Index Bootstrapping**: Creates the store with embeddings initialized
  through `init_embeddings`, mirroring the LangGraph docs recipe.
- **Targeted Memory Writes**: Stores user claims behind stable UUID keys only
  when the message explicitly requests persistence.
- **Contextual Prompting**: Injects retrieved memory snippets into the system
  message so replies stay personalized without replaying entire history.

=== COMPARISON WITH SIMPLE STORES ===
| Plain InMemoryStore | Semantic Store (this file) |
|---------------------|----------------------------|
| Exact-string search only | Embedding search captures paraphrases |
| No scoring metadata      | Returns distances for observability |
| Hard to scale facts      | Namespaces keep tenants isolated |

What You'll Learn
1. Configure an embedding-backed `InMemoryStore` for semantic memory search.
2. Write selective memories keyed by user ID and inspect retrieval metadata.
3. Feed retrieved items into the prompt to personalize downstream responses.

Lesson Flow
1. Initialize embeddings and create an indexed store plus conversational graph.
2. Inside the graph, search for semantically similar memories on each user turn.
3. Run a sample conversation demonstrating memory writes and semantic recall.
"""

from __future__ import annotations

import uuid
from typing import Iterable

from langchain.embeddings import init_embeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from src.langgraph_learning.utils import (
    create_llm,
    maybe_enable_langsmith,
    pretty_print_messages,
    require_llm_provider_api_key,
    save_graph_image,
)


def create_semantic_store(model_name: str = "openai:text-embedding-3-small") -> InMemoryStore:
    """Create an embedding-backed store suitable for semantic recall."""
    embeddings = init_embeddings(model_name)
    return InMemoryStore(
        index={
            "embed": embeddings,
            "dims": embeddings.dimensions,
        }
    )


def build_semantic_graph(store: InMemoryStore):
    """Compile a LangGraph that injects semantic memories into the prompt."""
    llm = create_llm()
    memory = MemorySaver()

    def _namespace(config: RunnableConfig) -> tuple[str, str]:
        user_id = config["configurable"]["user_id"]
        return ("semantic-memory", user_id)

    def conversation(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
        namespace = _namespace(config)
        query = state["messages"][-1].content
        items = store.search(namespace, query=query, limit=3)
        memory_lines = []
        for item in items:
            distance = item.score if hasattr(item, "score") else None
            distance_txt = f" (score={distance:.3f})" if distance is not None else ""
            memory_lines.append(f"- {item.value['text']}{distance_txt}")
        memory_block = "\n".join(memory_lines) or "No stored memory found."
        system_msg = SystemMessage(
            content=f"You are a helpful assistant. Known facts:\n{memory_block}"
        )
        response = llm.invoke([system_msg, *state["messages"]])
        return {"messages": response}

    def write_memory(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
        namespace = _namespace(config)
        last = state["messages"][-1]
        text = last.content
        if "remember" not in text.lower():
            return {}
        identifier = uuid.uuid4().hex
        store.put(namespace, identifier, {"text": text})
        return {}

    builder = StateGraph(MessagesState)
    builder.add_node("conversation", conversation)
    builder.add_node("write_memory", write_memory)
    builder.add_edge(START, "conversation")
    builder.add_edge("conversation", "write_memory")
    builder.add_edge("write_memory", END)

    graph = builder.compile(store=store, checkpointer=memory)
    save_graph_image(graph, filename="artifacts/agent_with_semantic_memory.png")
    return graph


def run_demo(graph, prompts: Iterable[str]) -> None:
    """Stream a scripted conversation that triggers memory writes."""
    config = {"configurable": {"thread_id": "semantic-thread", "user_id": "semantic-user"}}
    for text in prompts:
        events = graph.stream(
            {"messages": [HumanMessage(content=text)]},
            config,
            stream_mode="values",
        )
        for event in events:
            pretty_print_messages(
                event["messages"][-1:], header=f"Assistant reply to: {text}"
            )


def main() -> None:
    require_llm_provider_api_key()
    maybe_enable_langsmith()
    store = create_semantic_store()
    graph = build_semantic_graph(store)
    prompts = [
        "Remember that I am a software engineer who loves trail running.",
        "Remember my favorite trail is the Dipsea in Marin.",
        "Suggest a new running route based on what you know about me.",
        "What's my profession?",
    ]
    run_demo(graph, prompts)


if __name__ == "__main__":
    main()

