"""
Advanced Memory System: Structured User Profiles with LangChain Structured Output

=== WHY THIS COURSE MATTERS ===
Getting LLMs to output structured data according to a schema is surprisingly difficult.
Traditional approaches face several challenges:

1. **Schema Compliance**: LLMs often produce malformed JSON or miss required fields
2. **Data Loss**: Regenerating entire schemas from scratch loses existing information
3. **Token Inefficiency**: Full rewrites waste tokens on unchanged data
4. **Complex Schema Failure**: Nested structures frequently cause parsing errors

LangChain's structured output solves these problems with automatic strategy selection,
ensuring structured output while providing reliable schema validation.

=== Core Concepts ===
1. Structured Memory: Use Pydantic Schema to define user profiles
2. Native Integration: LangChain automatically selects ProviderStrategy or ToolStrategy
3. Personalized Responses: Integrate memory into conversation context

=== Key Innovation ===
LangChain's structured output provides automatic strategy selection:
- ProviderStrategy for models with native structured output support (OpenAI, Grok)
- ToolStrategy for universal compatibility with all tool-calling models
- Built-in error handling and validation

=== Learning Objectives ===
1. Understand LangChain's structured output strategies
2. Learn to integrate structured memory into LangGraph
3. Compare different structured output approaches
"""

from __future__ import annotations

from functools import partial
from typing import List, Optional

from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
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

from src.langgraph_learning.utils import (
    create_llm,
    llm_from_config,
    maybe_enable_langsmith,
    pretty_print_messages,
    require_llm_provider_api_key,
    save_graph_image,
)


# =============================================================================
# 1. DEFINE THE USER PROFILE SCHEMA
# This is where we define what structured data we want to capture
# =============================================================================


class UserProfile(BaseModel):
    """Structured user profile that will be incrementally updated."""

    name: Optional[str] = Field(description="The user's name", default=None)
    location: Optional[str] = Field(description="Where the user lives", default=None)
    job: Optional[str] = Field(description="The user's occupation", default=None)
    interests: List[str] = Field(
        description="User's hobbies and interests", default_factory=list
    )


# =============================================================================
# 2. CORE WORKFLOW FUNCTIONS
# These implement the conversation and memory update flow
# =============================================================================

# System prompts for different tasks
MODEL_SYSTEM_MESSAGE = """You are a helpful assistant with memory.
Current user profile:
{profile}
Use this information to personalize your responses."""

STRUCTURED_OUTPUT_INSTRUCTION = """Reflect on the conversation and update the user profile.
Only include details directly stated by the user."""


def format_profile_for_display(profile: Optional[dict]) -> str:
    """Format profile data for display in system prompts."""
    if not profile:
        return "No profile information available."

    name = profile.get("name") or "Unknown"
    location = profile.get("location") or "Unknown"
    job = profile.get("job") or "Unknown"
    interests = ", ".join(profile.get("interests", [])) or "None listed"

    return f"Name: {name}\nLocation: {location}\nJob: {job}\nInterests: {interests}"


def _call_model_with_memory(
    state: MessagesState,
    config: RunnableConfig,
    store: BaseStore,
    *,
    llm: ChatOpenAI,
):
    """
    Process user message with personalized context from memory.

    This node:
    1. Loads the current user profile from memory
    2. Formats it for the system prompt
    3. Generates a personalized response
    """
    user_id = config["configurable"]["user_id"]
    namespace = ("profile", user_id)

    # Load existing profile
    existing_memory = store.get(namespace, "user_profile")
    current_profile = existing_memory.value if existing_memory else None

    # Format profile for display
    formatted_profile = format_profile_for_display(current_profile)

    # Create personalized system message
    system_msg = SystemMessage(
        content=MODEL_SYSTEM_MESSAGE.format(profile=formatted_profile)
    )

    # Generate response with memory context
    response = llm.invoke([system_msg, *state["messages"]])

    return {"messages": response}


def _update_profile_with_structured_output(
    state: MessagesState,
    config: RunnableConfig,
    store: BaseStore,
    *,
    structured_agent,
):
    """
    Update user profile using LangChain's structured output.

    KEY INNOVATION: LangChain automatically selects the best strategy
    (ProviderStrategy for native support, ToolStrategy for universal compatibility)
    and provides built-in error handling and validation.
    """
    user_id = config["configurable"]["user_id"]
    namespace = ("profile", user_id)

    # Load existing profile
    existing_memory = store.get(namespace, "user_profile")
    current_profile = existing_memory.value if existing_memory else None

    # Format existing profile for context
    existing_context = (
        f"Current profile: {current_profile}"
        if current_profile
        else "No existing profile"
    )

    # Use structured agent to extract and update profile
    result = structured_agent.invoke(
        {
            "messages": [
                SystemMessage(
                    content=f"{STRUCTURED_OUTPUT_INSTRUCTION}\n\n{existing_context}"
                ),
                *state["messages"],
            ]
        }
    )

    # Extract and store the updated profile
    structured_response = result.get("structured_response")
    if structured_response:
        updated_profile = structured_response.model_dump()
        store.put(namespace, "user_profile", updated_profile)


# =============================================================================
# 3. GRAPH CONSTRUCTION
# This builds the complete memory-enabled conversation system
# =============================================================================


def _assemble_profile_graph(
    llm: ChatOpenAI,
    extractor_llm: ChatOpenAI,
    *,
    store: BaseStore | None = None,
    checkpointer: MemorySaver | None = None,
):
    """
    Build a LangGraph that maintains structured user profiles.

    Graph Flow:
    START → call_model_with_memory → update_profile_with_structured_output → END

    Each conversation turn:
    1. Responds using current memory
    2. Updates memory with new information using structured output
    """
    response_llm = llm

    # Create structured agent for profile extraction
    structured_agent = create_agent(
        model=extractor_llm,
        tools=[],
        response_format=ToolStrategy(schema=UserProfile, handle_errors=True),
    )

    store = store or InMemoryStore()
    checkpointer = checkpointer or MemorySaver()

    builder = StateGraph(MessagesState)
    builder.add_node("assistant", partial(_call_model_with_memory, llm=response_llm))
    builder.add_node(
        "update_profile",
        partial(
            _update_profile_with_structured_output, structured_agent=structured_agent
        ),
    )

    # Define flow
    builder.add_edge(START, "assistant")
    builder.add_edge("assistant", "update_profile")
    builder.add_edge("update_profile", END)

    return builder.compile(store=store, checkpointer=checkpointer)


def build_profile_graph(
    model: ChatOpenAI | None = None,
    *,
    extractor_model: ChatOpenAI | None = None,
    store: BaseStore | None = None,
    checkpointer: MemorySaver | None = None,
):
    llm = model or create_llm()
    extractor_llm = extractor_model or create_llm()
    graph = _assemble_profile_graph(
        llm,
        extractor_llm,
        store=store,
        checkpointer=checkpointer,
    )
    save_graph_image(
        graph, filename="artifacts/agent_with_structured_memory.png", xray=True
    )
    return graph


# =============================================================================
# 4. DEMONSTRATION AND USAGE
# Show how the system works in practice
# =============================================================================


def demo_conversation(graph) -> None:
    """Run a short conversation that demonstrates structured memory with LangChain structured output."""
    config = {"configurable": {"thread_id": "demo", "user_id": "user123"}}

    print("=== DEMONSTRATION: Structured Memory with LangChain Structured Output ===\n")

    # Conversation 1: Create initial profile
    print("1. First conversation - Creating profile:")
    print("   User: 'Hi, I'm Leslie. I live in Shanghai and work as a engineer.'")

    result1 = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    "Hi, I'm Leslie. I live in Shanghai and work as a engineer."
                )
            ]
        },
        config,
    )

    print(f"   Assistant: '{result1['messages'][-1].content}'")

    # Check stored profile
    store: InMemoryStore = graph.store
    profile1 = store.get(("profile", "user123"), "user_profile")
    print(f"   Stored Profile: {profile1.value}\n")

    # Conversation 2: Incremental update
    print("2. Second conversation - Incremental update:")
    print("   User: 'I really enjoy painting and photography.'")

    result2 = graph.invoke(
        {"messages": [HumanMessage("I really enjoy painting and photography.")]}, config
    )

    print(f"   Assistant: '{result2['messages'][-1].content}'")

    # Check updated profile
    profile2 = store.get(("profile", "user123"), "user_profile")
    print(f"   Updated Profile: {profile2.value}\n")

    # Conversation 3: Using memory for personalization
    print("3. Third conversation - Using memory:")
    print("   User: 'What art galleries do you recommend?'")

    result3 = graph.invoke(
        {"messages": [HumanMessage("What art galleries do you recommend?")]}, config
    )

    print(f"   Assistant: '{result3['messages'][-1].content}'")

    print("\n=== KEY TAKEAWAY ===")
    print(
        "LangChain's structured output provides reliable schema validation with automatic strategy selection."
    )
    print(
        "ProviderStrategy for native support, ToolStrategy for universal compatibility."
    )


def main() -> None:
    require_llm_provider_api_key()
    maybe_enable_langsmith()
    graph = build_profile_graph()
    demo_conversation(graph)


if __name__ == "__main__":
    main()


def studio_graph(config: RunnableConfig | None = None):
    """Studio entry point for the structured memory lesson."""
    llm, overrides = llm_from_config(config)
    extractor_llm = create_llm(
        provider=overrides.get("extractor_provider") or overrides.get("provider"),
        model=overrides.get("extractor_model") or overrides.get("model"),
        temperature=(
            overrides.get("extractor_temperature")
            if overrides.get("extractor_temperature") is not None
            else overrides.get("temperature")
        ),
        api_key=overrides.get("extractor_api_key") or overrides.get("api_key"),
        base_url=overrides.get("extractor_base_url") or overrides.get("base_url"),
    )
    store = InMemoryStore()
    checkpointer = MemorySaver()
    return _assemble_profile_graph(
        llm,
        extractor_llm,
        store=store,
        checkpointer=checkpointer,
    )
