"""
Advanced Memory System: Flexible Fact Collection with TrustCall

=== PROBLEM STATEMENT ===
Traditional memory systems often force all user information into rigid schemas, but real conversations contain diverse facts that don't fit predefined categories.
How can we capture and manage multiple independent observations about users without being constrained by fixed field structures?

=== CORE SOLUTION ===
Instead of maintaining a single structured profile, this system accumulates multiple discrete facts as a searchable collection.
Each fact is stored independently, allowing flexible accumulation of diverse user information over time.

=== KEY INNOVATION ===
- **Flexible Schema**: Simple Memory schema with just a 'content' field
- **Multiple Storage**: Each fact stored as separate document with unique key
- **Searchable Collection**: All memories can be retrieved and searched independently
- **TrustCall Inserts**: Enable insertion of new memories alongside updates

=== COMPARISON WITH STRUCTURED PROFILES ===
| Structured Profiles (agent_with_structured_memory.py) | Fact Collections (this file) |
|------------------------------------------------------|-----------------------------|
| Single document per user | Multiple documents per user |
| Fixed field schema | Flexible content-only schema |
| Profile management | Fact accumulation |
| Incremental updates | Insert/update multiple items |

What You'll Learn
1. Model user memories as flexible Pydantic schemas and store them as searchable collections.
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

from src.langgraph_learning.stage05_advanced_memory_systems.configuration import (
    MemoryConfiguration,
)
from src.langgraph_learning.utils import (
    create_llm,
    maybe_enable_langsmith,
    pretty_print_messages,
    require_llm_provider_api_key,
    save_graph_image,
)


class Memory(BaseModel):
    """Simple schema describing a single fact about the user.

    KEY INNOVATION: Unlike structured profiles with fixed fields, this flexible
    schema allows capturing diverse facts without predefined categories.
    """

    content: str = Field(
        description="The main content of the memory. Example: User enjoys biking around San Francisco."
    )


MODEL_SYSTEM_MESSAGE = """You are a helpful companion that remembers facts about the user.
Here is your current memory (may be empty):
{memory}"""

TRUSTCALL_INSTRUCTION = """Reflect on this interaction.
Use the Memory tool to retain any useful details about the user.
Insert brand new memories or update existing ones as needed."""


def _create_config_helper(config: RunnableConfig) -> MemoryConfiguration:
    """Helper function to create MemoryConfiguration from RunnableConfig."""
    return MemoryConfiguration.from_runnable_config(config)


def _create_read_facts_node(llm: ChatOpenAI):
    """Create the node that reads multiple facts from memory collection.

    KEY DIFFERENCE: Unlike single-profile systems, this searches ALL stored facts
    and formats them as a collection for the system prompt.
    """

    def read_facts(state: MessagesState, config: RunnableConfig, store: BaseStore):
        """Process user message with memory context from stored fact collection.

        CORE CONCEPT: This demonstrates the power of fact collections -
        we can retrieve and use ALL stored facts, not just a single profile.

        Args:
            state: Current conversation state
            config: Runnable configuration
            store: Memory store for user facts

        Returns:
            Updated state with model response
        """
        cfg = _create_config_helper(config)
        namespace = ("memories", cfg.user_id)
        memories = list(store.search(namespace))

        # Format stored memories for system prompt
        formatted = (
            "\n".join(f"- {item.value['content']}" for item in memories)
            or "No saved memories."
        )

        system_msg = MODEL_SYSTEM_MESSAGE.format(memory=formatted)
        response = llm.invoke([SystemMessage(content=system_msg), *state["messages"]])
        return {"messages": response}

    return read_facts


def _create_update_facts_node(llm: ChatOpenAI):
    """Create the node that updates fact collection using TrustCall.

    KEY INNOVATION: TrustCall with enable_inserts=True allows creating
    NEW facts alongside updating existing ones - perfect for collections.
    """
    extractor = create_extractor(
        llm,
        tools=[Memory],
        tool_choice="Memory",
        enable_inserts=True,  # CRITICAL: Allows inserting new facts
    )

    def update_facts(state: MessagesState, config: RunnableConfig, store: BaseStore):
        """Update fact collection using TrustCall extractor.

        CORE CONCEPT: Each fact is stored as a separate document with unique key.
        This enables flexible accumulation of diverse user information over time.

        Args:
            state: Current conversation state
            config: Runnable configuration
            store: Memory store for user facts
        """
        cfg = _create_config_helper(config)
        namespace = ("memories", cfg.user_id)
        current_items = list(store.search(namespace))

        # Prepare existing facts for TrustCall - each gets its own key
        existing = (
            [(item.key, "Memory", item.value) for item in current_items]
            if current_items
            else None
        )

        # Invoke TrustCall extractor to update fact collection
        request_messages = [
            SystemMessage(content=TRUSTCALL_INSTRUCTION),
            *state["messages"],
        ]
        result = extractor.invoke({"messages": request_messages, "existing": existing})

        # Store updated facts - each as separate document
        for response, meta in zip(result["responses"], result["response_metadata"]):
            key = meta.get("json_doc_id") or str(uuid.uuid4())
            store.put(namespace, key, response.model_dump(mode="json"))

    return update_facts


def build_agent_with_fact_collection(model: ChatOpenAI | None = None):
    """Compile a graph that stores multiple user facts via TrustCall.

    GRAPH FLOW: START → read_facts → update_facts → END

    Each conversation turn:
    1. Reads ALL stored facts for personalized context
    2. Updates fact collection (inserts new facts, updates existing ones)

    KEY DIFFERENCE: Unlike single-profile systems, this maintains a COLLECTION
    of independent facts, each stored as a separate document.

    Args:
        model: Optional ChatOpenAI model instance. If None, creates default LLM.

    Returns:
        Compiled LangGraph with fact collection capabilities
    """
    llm = model or create_llm()

    # Create node functions
    read_facts = _create_read_facts_node(llm)
    update_facts = _create_update_facts_node(llm)

    # Build the graph
    builder = StateGraph(MessagesState, config_schema=MemoryConfiguration)
    builder.add_node("read_facts", read_facts)
    builder.add_node("update_facts", update_facts)
    builder.add_edge(START, "read_facts")
    builder.add_edge("read_facts", "update_facts")
    builder.add_edge("update_facts", END)

    graph = builder.compile(store=InMemoryStore(), checkpointer=MemorySaver())
    save_graph_image(
        graph, filename="artifacts/agent_with_fact_collection.png", xray=True
    )
    return graph


def inspect_saved_facts(graph, user_id: str) -> None:
    """Inspect and display all saved facts for a given user.

    DEMONSTRATION: Shows the power of fact collections - multiple independent
    facts stored separately, each with its own unique key.

    Args:
        graph: Compiled LangGraph instance
        user_id: User identifier to retrieve facts for
    """
    memory_store: InMemoryStore = graph.store  # type: ignore[assignment]
    namespace = ("memories", user_id)
    items = list(memory_store.search(namespace))

    if not items:
        print("No facts stored yet.")
        return

    print(f"\n=== FACT COLLECTION FOR USER: {user_id} ===")
    print(f"Total facts stored: {len(items)}")
    print("\nIndividual facts:")
    for item in items:
        print(f"  [{item.key}] {item.value['content']}")
    print("\n=== KEY TAKEAWAY ===")
    print(
        "Each fact is stored independently, allowing flexible accumulation over time."
    )


def main() -> None:
    require_llm_provider_api_key()
    maybe_enable_langsmith()
    graph = build_agent_with_fact_collection()
    config = {
        "configurable": {"thread_id": "fact-collection-demo", "user_id": "leslie"}
    }

    print("=== DEMONSTRATION: Flexible Fact Collection ===\n")

    # Demonstrate accumulation of diverse facts
    turns = [
        "Hi, I'm Leslie.",
        "I live in Shanghai and love biking.",
        "I also enjoy reading books.",
        "My favorite coffee shop is Blue Bottle.",
    ]

    for text in turns:
        print(f"User: '{text}'")
        chunk = graph.invoke({"messages": [HumanMessage(content=text)]}, config)
        print(f"Assistant: '{chunk['messages'][-1].content}'\n")

    # Show the accumulated fact collection
    inspect_saved_facts(graph, user_id="leslie")


if __name__ == "__main__":
    main()
