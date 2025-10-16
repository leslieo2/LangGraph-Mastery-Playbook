"""This LangGraph memory service extracts structured facts with TrustCall, stores them as a searchable collection, and replays them during conversation.

What You'll Learn
1. Model user memories as Pydantic schemas and store them as a searchable collection.
2. Invoke TrustCall to insert or update memories in parallel with structured outputs.
3. Compile a LangGraph that reads existing memories, updates them, and logs the results.

Lesson Flow
1. Define the `Memory` schema and create a TrustCall extractor configured for inserts.
2. Build read/write nodes that render memory into prompts and persist TrustCall responses.
3. Run scripted interactions, then iterate through the memory store to inspect saved items.
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel, Field

from trustcall import create_extractor

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

from src.langgraph_learning.stage02_memory_and_personalization.configuration import (
    MemoryConfiguration,
)
from src.langgraph_learning.utils import (
    create_llm,
    maybe_enable_langsmith,
    pretty_print_messages,
    require_env,
    save_graph_image,
)


class Memory(BaseModel):
    """Simple schema describing a single fact about the user."""

    content: str = Field(
        description="The main content of the memory. Example: User enjoys biking around San Francisco."
    )


MODEL_SYSTEM_MESSAGE = """You are a helpful companion that remembers facts about the user.
Here is your current memory (may be empty):
{memory}"""

TRUSTCALL_INSTRUCTION = """Reflect on this interaction.
Use the Memory tool to retain any useful details about the user.
Insert brand new memories or update existing ones as needed."""


def build_memory_collection_graph(model: ChatOpenAI | None = None):
    """Compile a graph that stores multiple user memories via TrustCall."""
    llm = model or create_llm()
    extractor = create_extractor(
        llm,
        tools=[Memory],
        tool_choice="Memory",
        enable_inserts=True,
    )

    def _config(config: RunnableConfig) -> MemoryConfiguration:
        return MemoryConfiguration.from_runnable_config(config)

    def call_model(state: MessagesState, config: RunnableConfig, store: BaseStore):
        cfg = _config(config)
        namespace = ("memories", cfg.user_id)
        memories = list(store.search(namespace))
        formatted = (
            "\n".join(f"- {item.value['content']}" for item in memories)
            or "No saved memories."
        )
        system_msg = MODEL_SYSTEM_MESSAGE.format(memory=formatted)
        response = llm.invoke([SystemMessage(content=system_msg), *state["messages"]])
        return {"messages": response}

    def write_memory(state: MessagesState, config: RunnableConfig, store: BaseStore):
        cfg = _config(config)
        namespace = ("memories", cfg.user_id)
        current_items = list(store.search(namespace))
        existing = (
            [(item.key, "Memory", item.value) for item in current_items]
            if current_items
            else None
        )
        request_messages = [
            SystemMessage(content=TRUSTCALL_INSTRUCTION),
            *state["messages"],
        ]
        result = extractor.invoke({"messages": request_messages, "existing": existing})

        for response, meta in zip(result["responses"], result["response_metadata"]):
            key = meta.get("json_doc_id") or str(uuid.uuid4())
            store.put(namespace, key, response.model_dump(mode="json"))

    builder = StateGraph(MessagesState, config_schema=MemoryConfiguration)
    builder.add_node("call_model", call_model)
    builder.add_node("write_memory", write_memory)
    builder.add_edge(START, "call_model")
    builder.add_edge("call_model", "write_memory")
    builder.add_edge("write_memory", END)

    graph = builder.compile(store=InMemoryStore(), checkpointer=MemorySaver())
    save_graph_image(graph, filename="artifacts/memory_collection_graph.png", xray=True)
    return graph


def inspect_saved_memories(graph, user_id: str) -> None:
    memory_store: InMemoryStore = graph.store  # type: ignore[assignment]
    namespace = ("memories", user_id)
    items = list(memory_store.search(namespace))
    if not items:
        print("No memories stored yet.")
    for item in items:
        print(f"[{item.key}] {item.value['content']}")


def main() -> None:
    require_env("OPENAI_API_KEY")
    maybe_enable_langsmith()
    graph = build_memory_collection_graph()
    config = {"configurable": {"thread_id": "thread-collection", "user_id": "leslie"}}
    turns = [
        "Hi, I'm Leslie.",
        "I live in San Francisco and love biking.",
        "I also enjoy hunting for the best croissant in town.",
    ]
    for text in turns:
        chunk = graph.invoke({"messages": [HumanMessage(content=text)]}, config)
        pretty_print_messages(
            chunk["messages"][-1:], header=f"Assistant reply to: {text}"
        )

    inspect_saved_memories(graph, user_id="leslie")


if __name__ == "__main__":
    main()
