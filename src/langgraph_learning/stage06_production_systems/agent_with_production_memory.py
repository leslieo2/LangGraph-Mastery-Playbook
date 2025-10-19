"""
Production Memory Systems: Durable Backends for LangGraph

=== PROBLEM STATEMENT ===
In-memory checkpoints vanish when a process restarts, and SQLite struggles once
multiple workers share the same agent. Production deployments need durable,
multi-tenant persistence that scales across machines.

=== CORE SOLUTION ===
This lesson wires a personalization agent to Postgres, MongoDB, and Redis
backends. The graph reads user memory before replying, then writes updates
through the appropriate store and checkpointer context managers.

=== KEY INNOVATION ===
- **Configurable Backends**: One builder function works with Postgres, MongoDB,
  or Redis, so teams can pick an existing infrastructure stack.
- **First-Run Setup Hooks**: Helpers expose `.setup()` / `.asetup()` so schema
  creation only runs on demand.
- **Sync + Async Pathways**: Async factory callables mirror the sync API, making
  it easy to upgrade to non-blocking agents.

=== COMPARISON WITH IN-MEMORY CHECKPOINTS ===
| MemorySaver / InMemoryStore | Production Backends (this file) |
|-----------------------------|----------------------------------|
| Data lost after process exit | Records durably stored in external DBs |
| Single-process only          | Safe for multi-worker deployments |
| No operational hooks         | Explicit setup, teardown, and monitoring |

What You'll Learn
1. Initialize LangGraph memory using Postgres, MongoDB, or Redis connectors.
2. Share one graph builder across backends by injecting store + checkpointer.
3. Stream a demo conversation while persisting user-specific context.

Lesson Flow
1. Define backend helpers that open the correct store/checkpointer pair and
   expose setup methods.
2. Compile a personalization graph that reads long-term memory and writes
   bullet updates after each reply.
3. Run a short demo that proves memory survives across multiple turns.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Protocol

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.store.base import BaseStore

from src.langgraph_learning.utils import (
    create_llm,
    maybe_enable_langsmith,
    pretty_print_messages,
    require_llm_provider_api_key,
    save_graph_image,
)


class HasSetup(Protocol):
    """Minimal protocol for stores/savers that expose setup routines."""

    def setup(self) -> None: ...


class HasAsyncSetup(Protocol):
    """Protocol for async stores/savers that expose asetup routines."""

    async def asetup(self) -> None: ...


StoreFactory = Callable[[str, bool], Iterator[tuple[BaseStore, Any]]]


@dataclass(slots=True)
class Backend:
    """Metadata describing how to load memory backends."""

    name: str
    sync_factory: StoreFactory
    async_factory: str | None  # Documented for travelers using async graphs.


@contextmanager
def postgres_memory(uri: str, initialize: bool = False):
    """Yield Postgres store + checkpointer pair."""
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        from langgraph.store.postgres import PostgresStore
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "Install `langgraph-checkpoint-postgres` to enable Postgres persistence."
        ) from exc

    with (
        PostgresStore.from_conn_string(uri) as store,
        PostgresSaver.from_conn_string(uri) as saver,
    ):
        if initialize:
            store.setup()
            saver.setup()
        yield store, saver


@contextmanager
def mongodb_memory(uri: str, initialize: bool = False):
    """Yield MongoDB store + checkpointer pair."""
    try:
        from langgraph.checkpoint.mongodb import MongoDBSaver
        from langgraph.store.mongodb import MongoDBStore
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "Install `langgraph-checkpoint-mongodb` to enable MongoDB persistence."
        ) from exc

    with (
        MongoDBStore.from_conn_string(uri) as store,
        MongoDBSaver.from_conn_string(uri) as saver,
    ):
        if initialize:
            store.setup()
            saver.setup()
        yield store, saver


@contextmanager
def redis_memory(uri: str, initialize: bool = False):
    """Yield Redis store + checkpointer pair."""
    try:
        from langgraph.checkpoint.redis import RedisSaver
        from langgraph.store.redis import RedisStore
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "Install `langgraph-checkpoint-redis` to enable Redis persistence."
        ) from exc

    with (
        RedisStore.from_conn_string(uri) as store,
        RedisSaver.from_conn_string(uri) as saver,
    ):
        if initialize:
            store.setup()
            saver.setup()
        yield store, saver


BACKENDS: dict[str, Backend] = {
    "postgres": Backend(
        name="Postgres",
        sync_factory=postgres_memory,
        async_factory="langgraph.checkpoint.postgres.aio.AsyncPostgresSaver",
    ),
    "mongodb": Backend(
        name="MongoDB",
        sync_factory=mongodb_memory,
        async_factory="langgraph.checkpoint.mongodb.aio.AsyncMongoDBSaver",
    ),
    "redis": Backend(
        name="Redis",
        sync_factory=redis_memory,
        async_factory="langgraph.checkpoint.redis.aio.AsyncRedisSaver",
    ),
}


def build_personalization_graph(store: BaseStore, saver: Any):
    """Compile a graph that personalizes responses and writes bullet memories."""
    llm = create_llm()

    def call_model(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
        cfg = config["configurable"]
        namespace = ("profile", cfg["user_id"])
        memory = store.get(namespace, "user_memory")
        prior = memory.value["notes"] if memory else "No prior memory."
        system_msg = SystemMessage(
            content=f"You are a durable assistant. User notes:\n{prior}"
        )
        response = llm.invoke([system_msg, *state["messages"]])
        return {"messages": response}

    def write_memory(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
        cfg = config["configurable"]
        namespace = ("profile", cfg["user_id"])
        latest = state["messages"][-1].content
        history = store.get(namespace, "user_memory")
        earlier = history.value["notes"] if history else ""
        merged = "\n".join(filter(None, [earlier, f"- {latest}"]))
        store.put(namespace, "user_memory", {"notes": merged})

    builder = StateGraph(MessagesState)
    builder.add_node("assistant", call_model)
    builder.add_node("write_memory", write_memory)
    builder.add_edge(START, "assistant")
    builder.add_edge("assistant", "write_memory")
    builder.add_edge("write_memory", END)

    graph = builder.compile(store=store, checkpointer=saver)
    save_graph_image(graph, filename="artifacts/agent_with_production_memory.png")
    return graph


def run_demo(graph) -> None:
    """Stream a short conversation to prove persistence across turns."""
    config = {
        "configurable": {
            "thread_id": "prod-thread-001",
            "user_id": "prod-user-42",
        }
    }
    prompts = [
        "Remember that my name is Leslie and I live in San Francisco.",
        "Make a note that I coach a weekend robotics club.",
        "What do you already know about me?",
    ]
    for text in prompts:
        chunk_stream = graph.stream(
            {"messages": [HumanMessage(content=text)]},
            config,
            stream_mode="values",
        )
        for chunk in chunk_stream:
            pretty_print_messages(
                chunk["messages"][-1:], header=f"Assistant reply to: {text}"
            )


def main() -> None:
    """
    Bootstraps the chosen backend and runs the demo conversation.

    Set BACKEND_URI and BACKEND_KIND environment variables before running:
    BACKEND_KIND=postgres BACKEND_URI=postgresql://... python agent_with_production_memory.py
    """

    require_llm_provider_api_key()
    maybe_enable_langsmith()

    backend_key = os.environ.get("BACKEND_KIND", "postgres").lower()
    backend = BACKENDS.get(backend_key)
    if not backend:
        raise ValueError(f"Unsupported BACKEND_KIND: {backend_key}")

    uri = os.environ.get("BACKEND_URI")
    if not uri:
        raise ValueError("Set BACKEND_URI to the database connection string.")

    initialize = os.environ.get("BACKEND_INITIALIZE", "false").lower() == "true"
    with backend.sync_factory(uri, initialize) as (store, saver):
        graph = build_personalization_graph(store, saver)
        run_demo(graph)


if __name__ == "__main__":
    main()
