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

from pydantic import BaseModel, Field, ValidationError

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

    Args:
        profile: User profile data as dict or None if no profile exists

    Returns:
        Formatted string representation of the profile
    """
    if not profile:
        return "No profile stored yet."

    if isinstance(profile, BaseModel):
        profile = profile.model_dump(mode="json")

    name = profile.get("name") or "Unknown"
    location = profile.get("location") or "Unknown"
    job = profile.get("job") or "Unknown"
    interests = ", ".join(profile.get("interests") or []) or "None listed"

    return f"Name: {name}\nLocation: {location}\nJob: {job}\nInterests: {interests}"


def _create_config_helper(config: RunnableConfig) -> MemoryConfiguration:
    """Helper function to create MemoryConfiguration from RunnableConfig."""
    return MemoryConfiguration.from_runnable_config(config)


def _create_call_model_node(llm: ChatOpenAI):
    """Create the call model node for processing user messages."""

    def call_model(state: MessagesState, config: RunnableConfig, store: BaseStore):
        """Process user message with personalized system prompt.

        Args:
            state: Current conversation state
            config: Runnable configuration
            store: Memory store for profile data

        Returns:
            Updated state with model response
        """
        cfg = _create_config_helper(config)
        namespace = ("profile", cfg.user_id)
        existing = store.get(namespace, "user_profile")
        stored_profile = None
        if existing:
            stored_profile = existing.value.get("profile")  # type: ignore[assignment]

        formatted_profile = _format_profile(stored_profile)
        response = llm.invoke(
            [
                SystemMessage(
                    content=MODEL_SYSTEM_MESSAGE.format(profile=formatted_profile)
                ),
                *state["messages"],
            ]
        )
        return {"messages": response}

    return call_model


def _prepare_existing_profile(existing: object | None) -> tuple[str, list | None]:
    """Prepare existing profile data for TrustCall extractor.

    Args:
        existing: Existing profile data from store

    Returns:
        Tuple of (doc_id, existing_payload) for TrustCall
    """
    existing_doc_id = "user_profile"
    existing_payload = None

    if existing:
        existing_doc_id = existing.value.get("doc_id", existing_doc_id)
        profile_payload = existing.value.get("profile") or {}

        if isinstance(profile_payload, BaseModel):
            profile_payload = profile_payload.model_dump(mode="json")
        elif not isinstance(profile_payload, dict):
            try:
                profile_payload = Profile.model_validate(profile_payload).model_dump(
                    mode="json"
                )
            except ValidationError:
                profile_payload = {}

        existing_payload = [
            (
                existing_doc_id,
                "Profile",
                profile_payload,
            )
        ]

    return existing_doc_id, existing_payload


def _process_profile_response(profile_response: object) -> dict | None:
    """Process TrustCall response to extract profile data.

    Args:
        profile_response: Response from TrustCall extractor

    Returns:
        Processed profile data as dict, or None if invalid
    """
    if isinstance(profile_response, BaseModel):
        return profile_response.model_dump(mode="json")
    elif isinstance(profile_response, dict):
        return profile_response
    else:
        try:
            return Profile.model_validate(profile_response).model_dump(mode="json")
        except ValidationError:
            return None


def _create_write_memory_node(llm: ChatOpenAI):
    """Create the write memory node for updating user profiles."""
    extractor = create_extractor(
        llm,
        tools=[Profile],
        tool_choice="Profile",
    )

    def write_memory(state: MessagesState, config: RunnableConfig, store: BaseStore):
        """Update user profile using TrustCall extractor.

        Args:
            state: Current conversation state
            config: Runnable configuration
            store: Memory store for profile data
        """
        cfg = _create_config_helper(config)
        namespace = ("profile", cfg.user_id)
        existing = store.get(namespace, "user_profile")

        # Prepare existing profile data for TrustCall
        existing_doc_id, existing_payload = _prepare_existing_profile(existing)

        # Invoke TrustCall extractor to update profile
        result = extractor.invoke(
            {
                "messages": [
                    SystemMessage(content=TRUSTCALL_INSTRUCTION),
                    *state["messages"],
                ],
                "existing": existing_payload,
            }
        )

        # Process extractor response
        responses = result.get("responses") or []
        if not responses:
            return

        profile_response = responses[0]
        updated_profile = _process_profile_response(profile_response)
        if not updated_profile:
            return

        # Store updated profile
        metadata = result.get("response_metadata") or [{}]
        doc_id = metadata[0].get("json_doc_id") or existing_doc_id
        store.put(
            namespace,
            "user_profile",
            {
                "doc_id": doc_id,
                "profile": updated_profile,
            },
        )

    return write_memory


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
    call_model = _create_call_model_node(llm)
    write_memory = _create_write_memory_node(llm)

    # Build the graph
    builder = StateGraph(MessagesState, context_schema=MemoryConfiguration)
    builder.add_node("assistant", call_model)
    builder.add_node("update_profile", write_memory)
    builder.add_edge(START, "assistant")
    builder.add_edge("assistant", "update_profile")
    builder.add_edge("update_profile", END)

    graph = builder.compile(store=InMemoryStore(), checkpointer=MemorySaver())
    save_graph_image(graph, filename="artifacts/profile_memory_graph.png", xray=True)
    return graph


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
