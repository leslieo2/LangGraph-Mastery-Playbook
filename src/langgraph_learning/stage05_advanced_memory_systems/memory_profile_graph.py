"""This LangGraph companion captures structured user profiles with TrustCall, personalizes model prompts, and keeps the profile up to date after each turn.

What You'll Learn
1. Capture structured user profiles using TrustCall and a Pydantic schema.
2. Prepend stored profile details to prompts so the model can personalize replies.
3. Update the profile after each turn and inspect the stored snapshot for verification.

Lesson Flow
1. Declare the `Profile` schema and configure a TrustCall extractor for profile updates.
2. Implement read/write nodes that format profile details and persist TrustCall output.
3. Compile the graph, render the visualization, run sample prompts, and print the saved profile.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from pydantic import BaseModel, Field, ValidationError

from trustcall import create_extractor

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
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


class Profile(BaseModel):
    """Structured description of the user we are conversing with."""

    name: str | None = Field(description="The user's name", default=None)
    location: str | None = Field(description="Where the user is located", default=None)
    job: str | None = Field(
        description="The user's job title or occupation", default=None
    )
    interests: list[str] = Field(
        description="Any hobbies or interests explicitly mentioned by the user",
        default_factory=list,
    )


MODEL_SYSTEM_MESSAGE = """You are a helpful companion with a persistent user profile.
Here is the current profile (may be empty):
{profile}"""

TRUSTCALL_INSTRUCTION = """Reflect on the conversation and update the user profile.
Only include details directly stated by the user, keeping each field concise."""


def _format_profile(profile: dict[str, object] | None) -> str:
    """Format profile data for display in system prompts.

    This function converts profile data into a human-readable string format
    suitable for inclusion in system prompts. It handles various input types
    and provides default values for missing fields.

    Args:
        profile: User profile data as dict, BaseModel, or None if no profile exists

    Returns:
        Formatted string representation of the profile
    """
    if not profile:
        return "No profile stored yet."

    # Convert BaseModel to dict if needed
    if isinstance(profile, BaseModel):
        profile = profile.model_dump(mode="json")

    # Extract profile fields with defaults
    name = profile.get("name") or "Unknown"
    location = profile.get("location") or "Unknown"
    job = profile.get("job") or "Unknown"
    interests = ", ".join(profile.get("interests") or []) or "None listed"

    return f"Name: {name}\nLocation: {location}\nJob: {job}\nInterests: {interests}"


def _ensure_profile_dict(profile: object | None) -> dict[str, object] | None:
    """Normalize profile-like inputs to a plain dict.

    This function handles various profile input types and converts them to
    a standardized dictionary format. It supports BaseModel instances,
    dictionaries, and attempts to validate other objects as Profile models.

    Args:
        profile: Profile data in various formats (BaseModel, dict, or other)

    Returns:
        Normalized profile as dict, or None if conversion fails
    """
    if profile is None:
        return None
    if isinstance(profile, BaseModel):
        return profile.model_dump(mode="json")
    if isinstance(profile, dict):
        return profile
    try:
        return Profile.model_validate(profile).model_dump(mode="json")
    except ValidationError:
        return None


def _create_config_helper(config: RunnableConfig) -> MemoryConfiguration:
    """Helper function to create MemoryConfiguration from RunnableConfig.

    This function extracts user configuration from the RunnableConfig
    and converts it to a MemoryConfiguration object for consistent
    access to user-specific settings.

    Args:
        config: RunnableConfig containing user configuration

    Returns:
        MemoryConfiguration object with user-specific settings
    """
    return MemoryConfiguration.from_runnable_config(config)


def _create_call_model_node(llm: ChatOpenAI):
    """Create the call model node for processing user messages.

    This node handles the main conversation processing by:
    1. Loading the current user profile from memory
    2. Formatting the profile for inclusion in the system prompt
    3. Invoking the LLM with personalized context
    4. Returning the model response

    Args:
        llm: ChatOpenAI model instance for processing messages

    Returns:
        Function that processes messages with personalized context
    """

    def call_model(state: MessagesState, config: RunnableConfig, store: BaseStore):
        """Process user message with personalized system prompt.

        This function integrates user profile data into the conversation context
        to provide personalized responses based on stored user information.

        Args:
            state: Current conversation state containing message history
            config: Runnable configuration with user-specific settings
            store: Memory store for accessing user profile data

        Returns:
            Updated state containing the model's response message
        """
        # Extract user configuration and namespace
        cfg = _create_config_helper(config)
        namespace = ("profile", cfg.user_id)

        # Load and format current profile
        _, stored_profile = _load_profile_snapshot(store, namespace)
        formatted_profile = _format_profile(stored_profile)

        # Create personalized system message and invoke LLM
        system_message = SystemMessage(
            content=MODEL_SYSTEM_MESSAGE.format(profile=formatted_profile)
        )
        response = llm.invoke([system_message, *state["messages"]])

        return {"messages": response}

    return call_model


def _load_profile_snapshot(
    store: BaseStore, namespace: tuple[str, str]
) -> tuple[str, dict[str, object] | None]:
    """Load the stored profile document id and payload (if any)."""
    existing = store.get(namespace, "user_profile")
    doc_id = "user_profile"
    if not existing:
        return doc_id, None

    existing_value = getattr(existing, "value", {}) or {}
    doc_id = existing_value.get("doc_id", doc_id)
    stored_profile = _ensure_profile_dict(existing_value.get("profile"))
    return doc_id, stored_profile


def _build_existing_payload(
    doc_id: str, profile_payload: dict[str, object] | None
) -> list[tuple[str, str, dict[str, object]]] | None:
    """Convert stored profile data to TrustCall's expected 'existing' payload."""
    if not profile_payload:
        return None
    return [
        (
            doc_id,
            "Profile",
            profile_payload,
        )
    ]


def _extract_profile_update(
    extractor,
    messages: Sequence[BaseMessage],
    existing_payload: list[tuple[str, str, dict[str, object]]] | None,
    fallback_doc_id: str,
) -> tuple[str, dict[str, object]] | None:
    """Call TrustCall and normalize the resulting profile update."""
    result = extractor.invoke(
        {
            "messages": [
                SystemMessage(content=TRUSTCALL_INSTRUCTION),
                *messages,
            ],
            "existing": existing_payload,
        }
    )

    responses: Iterable[object] = result.get("responses") or []
    first_response = next(iter(responses), None)
    updated_profile = _ensure_profile_dict(first_response)
    if not updated_profile:
        return None

    metadata = result.get("response_metadata") or [{}]
    doc_id = metadata[0].get("json_doc_id") or fallback_doc_id
    return doc_id, updated_profile


def _create_extractor_for_profile(llm: ChatOpenAI):
    """Create TrustCall extractor for profile updates.

    Returns:
        TrustCall extractor configured for Profile schema
    """
    return create_extractor(
        llm,
        tools=[Profile],
        tool_choice="Profile",
    )


def _process_profile_update(
    extractor, messages, existing_payload, fallback_doc_id
):
    """Process profile update using TrustCall extractor.

    Args:
        extractor: TrustCall extractor instance
        messages: Conversation messages
        existing_payload: Existing profile payload
        fallback_doc_id: Fallback document ID

    Returns:
        Tuple of (doc_id, updated_profile) or None if no update
    """
    return _extract_profile_update(
        extractor,
        messages,
        existing_payload,
        fallback_doc_id=fallback_doc_id,
    )


def _store_updated_profile(store, namespace, doc_id, updated_profile):
    """Store updated profile in the memory store.

    Args:
        store: Memory store instance
        namespace: Store namespace
        doc_id: Document ID
        updated_profile: Updated profile data
    """
    store.put(
        namespace,
        "user_profile",
        {
            "doc_id": doc_id,
            "profile": updated_profile,
        },
    )


def _create_write_memory_node(llm: ChatOpenAI):
    """Create the write memory node for updating user profiles."""
    extractor = _create_extractor_for_profile(llm)

    def write_memory(state: MessagesState, config: RunnableConfig, store: BaseStore):
        """Update user profile using TrustCall extractor.

        Args:
            state: Current conversation state
            config: Runnable configuration
            store: Memory store for profile data
        """
        cfg = _create_config_helper(config)
        namespace = ("profile", cfg.user_id)

        # Load existing profile data
        existing_doc_id, stored_profile = _load_profile_snapshot(store, namespace)
        existing_payload = _build_existing_payload(existing_doc_id, stored_profile)

        # Process profile update
        update = _process_profile_update(
            extractor,
            state["messages"],
            existing_payload,
            existing_doc_id,
        )

        # Store updated profile if available
        if update:
            doc_id, updated_profile = update
            _store_updated_profile(store, namespace, doc_id, updated_profile)

    return write_memory


def _create_graph_nodes(llm: ChatOpenAI):
    """Create all graph nodes for the profile memory system.

    Args:
        llm: ChatOpenAI model instance

    Returns:
        Tuple of (call_model_node, write_memory_node)
    """
    call_model = _create_call_model_node(llm)
    write_memory = _create_write_memory_node(llm)
    return call_model, write_memory


def _build_graph_structure(builder, call_model, write_memory):
    """Build the graph structure with nodes and edges.

    Args:
        builder: StateGraph builder instance
        call_model: Call model node function
        write_memory: Write memory node function
    """
    builder.add_node("assistant", call_model)
    builder.add_node("update_profile", write_memory)
    builder.add_edge(START, "assistant")
    builder.add_edge("assistant", "update_profile")
    builder.add_edge("update_profile", END)


def _initialize_graph_builder():
    """Initialize the StateGraph builder with proper configuration.

    Returns:
        StateGraph builder instance configured for MessagesState
    """
    return StateGraph(MessagesState, context_schema=MemoryConfiguration)


def _compile_and_save_graph(builder):
    """Compile the graph and save visualization.

    Args:
        builder: StateGraph builder instance

    Returns:
        Compiled LangGraph instance
    """
    graph = builder.compile(store=InMemoryStore(), checkpointer=MemorySaver())
    save_graph_image(graph, filename="artifacts/profile_memory_graph.png", xray=True)
    return graph


def build_profile_graph(model: ChatOpenAI | None = None):
    """Compile a graph that maintains a structured user profile.

    This graph captures structured user profiles with TrustCall, personalizes
    model prompts, and keeps the profile up to date after each turn.

    Args:
        model: Optional ChatOpenAI model instance. If None, creates default LLM.

    Returns:
        Compiled LangGraph with profile memory capabilities
    """
    llm = model or create_llm()

    # Create node functions
    call_model, write_memory = _create_graph_nodes(llm)

    # Build the graph
    builder = _initialize_graph_builder()
    _build_graph_structure(builder, call_model, write_memory)

    # Compile and return the graph
    return _compile_and_save_graph(builder)


def main() -> None:
    require_llm_provider_api_key()
    maybe_enable_langsmith()
    graph = build_profile_graph()
    config = {"configurable": {"thread_id": "profile-demo", "user_id": "leslie"}}
    prompts = [
        "Hi, I'm Leslie. I live in San Francisco and work as a product manager.",
        "I love cycling around the city and exploring new bakeries.",
        "Remind me to plan a biking trip to Marin next month.",
    ]
    for text in prompts:
        output = graph.invoke({"messages": [HumanMessage(content=text)]}, config)
        pretty_print_messages(
            output["messages"][-1:], header=f"Assistant reply to: {text}"
        )

    # Display the stored profile for confirmation.
    store: InMemoryStore = graph.store  # type: ignore[assignment]
    snapshot = store.get(("profile", "leslie"), "user_profile")
    if snapshot:
        print("\nStored profile:")
        stored_profile = snapshot.value.get("profile")  # type: ignore[assignment]
        print(_format_profile(stored_profile))


if __name__ == "__main__":
    main()
