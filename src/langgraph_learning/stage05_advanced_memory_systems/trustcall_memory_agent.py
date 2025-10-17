"""This LangGraph multi-memory agent routes user requests through TrustCall extractors to update profiles, todos, and instructions in parallel.

What You'll Learn
1. Coordinate multiple memory types (profile, todos, instructions) inside one LangGraph agent.
2. Use TrustCall extractors to insert or patch structured documents based on conversation turns.
3. Route tool decisions dynamically and summarize TrustCall changes for transparent logging.

Lesson Flow
1. Define Pydantic schemas for profiles and todos, plus a tool-selection `TypedDict`.
2. Build the Task Maestro node that binds to `UpdateMemory`, then create specialized updater nodes.
3. Compile the graph with both long-term and short-term memory, stream scripted interactions, and observe follow-up behavior.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Iterable, Literal, Optional, Sequence, TypedDict

from pydantic import BaseModel, Field
from trustcall import create_extractor

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, merge_message_runs
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
    ToolCallSpy,
    maybe_enable_langsmith,
    pretty_print_messages,
    require_llm_provider_api_key,
    save_graph_image,
    summarize_tool_calls,
)


class Profile(BaseModel):
    """Stored details about the user we are helping."""

    name: Optional[str] = Field(description="The user's name", default=None)
    location: Optional[str] = Field(description="Where the user lives", default=None)
    job: Optional[str] = Field(description="The user's job or role", default=None)
    connections: list[str] = Field(
        default_factory=list,
        description="Friends and family mentioned by the user.",
    )
    interests: list[str] = Field(
        default_factory=list,
        description="Hobbies or interests the user has shared.",
    )


class ToDo(BaseModel):
    """Structured tasks captured for the user."""

    task: str = Field(description="The task to be completed.")
    time_to_complete: Optional[int] = Field(
        description="Estimated time to complete the task in minutes.", default=None
    )
    deadline: Optional[datetime] = Field(
        description="When the task needs to be completed by (if applicable).",
        default=None,
    )
    solutions: list[str] = Field(
        description="Actionable suggestions or vendors to help complete the task.",
        default_factory=list,
        min_items=1,
    )
    status: Literal["not started", "in progress", "done", "archived"] = Field(
        default="not started", description="Current status of the task."
    )


class UpdateMemory(TypedDict):
    """Decision returned by the assistant about which memory bucket to modify."""

    update_type: Literal["user", "todo", "instructions"]


SYSTEM_PROMPT = """You are a companion who helps the user remember important details.

Long-term memory contains:
1. User profile
2. The user's ToDo list
3. Custom instructions for how to manage the ToDo list

Current user profile:
<user_profile>
{user_profile}
</user_profile>

Current ToDo list:
<todo>
{todo}
</todo>

Current instructions:
<instructions>
{instructions}
</instructions>

Decide which long-term memory should be updated with the latest user message.
Always confirm ToDo updates to the user. Stay natural in your responses."""

TRUSTCALL_INSTRUCTION = """Reflect on the conversation below.
Use the provided tools to retain any necessary memories about the user.
Run updates and insertions in parallel where possible.

System Time: {time}"""

CREATE_INSTRUCTIONS_PROMPT = """Reflect on the conversation and update the instructions
that guide how ToDo items should be created or modified.

Current instructions:
<current_instructions>
{current_instructions}
</current_instructions>
"""


def _create_config_helper(config: RunnableConfig) -> MemoryConfiguration:
    """Helper function to create MemoryConfiguration from RunnableConfig."""
    return MemoryConfiguration.from_runnable_config(config)


def _load_user_profile(store: BaseStore, user_id: str) -> dict | None:
    """Load user profile from memory store."""
    existing = store.search(("profile", user_id))
    return existing[0].value if existing else None


def _load_user_todos(store: BaseStore, user_id: str) -> list[dict]:
    """Load user todos from memory store."""
    return [memory.value for memory in store.search(("todo", user_id))]


def _load_user_instructions(store: BaseStore, user_id: str) -> str:
    """Load user instructions from memory store."""
    existing = store.get(("instructions", user_id), "user_instructions")
    return existing.value.get("memory", "") if existing else ""


def _prepare_existing_docs(
    store: BaseStore,
    namespace: tuple[str, str],
    schema_label: str,
) -> list[tuple[str, str, dict]] | None:
    """Return TrustCall-ready tuples for existing namespace documents."""
    existing = store.search(namespace)
    if not existing:
        return None
    return [(item.key, schema_label, item.value) for item in existing]


def _format_trustcall_messages(
    messages: Sequence[BaseMessage],
    instruction: str,
) -> list[BaseMessage]:
    """Prepend system instruction and merge consecutive runs for TrustCall."""
    merged = merge_message_runs(
        [SystemMessage(content=instruction), *messages[:-1]]
    )
    return list(merged)


def _store_trustcall_responses(
    store: BaseStore,
    namespace: tuple[str, str],
    responses: Iterable[BaseModel],
    metadata: Iterable[dict],
) -> None:
    """Persist TrustCall responses back into the namespace."""
    for response, meta in zip(responses, metadata):
        key = meta.get("json_doc_id") or str(uuid.uuid4())
        store.put(namespace, key, response.model_dump(mode="json"))


def _last_tool_call_id(state: MessagesState) -> Optional[str]:
    """Safely extract the most recent tool call id from state messages."""
    message = state["messages"][-1]
    tool_calls = getattr(message, "tool_calls", None) or []
    if not tool_calls:
        return None
    return tool_calls[0].get("id")


def _tool_message(content: str, tool_call_id: Optional[str]) -> dict:
    """Format a single tool response payload."""
    payload: dict = {"role": "tool", "content": content}
    if tool_call_id:
        payload["tool_call_id"] = tool_call_id
    return payload


def _load_memory_data(store: BaseStore, user_id: str):
    """Load all memory data for the user.

    Args:
        store: Memory store instance
        user_id: User identifier

    Returns:
        Tuple of (profile, todos, instructions)
    """
    profile = _load_user_profile(store, user_id) or {}
    todos = _load_user_todos(store, user_id)
    instructions = _load_user_instructions(store, user_id)
    return profile, todos, instructions


def _format_todo_list(todos: list[dict]) -> str:
    """Format todo list for system prompt.

    Args:
        todos: List of todo items

    Returns:
        Formatted todo text
    """
    return "\n".join(map(str, todos)) or "No tasks saved."


def _create_system_message(profile: dict, todos: list[dict], instructions: str) -> SystemMessage:
    """Create system message with current memory state.

    Args:
        profile: User profile data
        todos: Todo list
        instructions: Custom instructions

    Returns:
        SystemMessage with formatted memory content
    """
    todo_text = _format_todo_list(todos)
    system_content = SYSTEM_PROMPT.format(
        user_profile=profile or "No profile stored.",
        todo=todo_text,
        instructions=instructions or "No custom instructions.",
    )
    return SystemMessage(content=system_content)


def _create_task_maistro_node(llm: ChatOpenAI):
    """Create the task maistro node for processing user requests."""

    def task_maistro(state: MessagesState, config: RunnableConfig, store: BaseStore):
        """Process user request and decide which memory to update."""
        cfg = _create_config_helper(config)

        # Load memory data
        profile, todos, instructions = _load_memory_data(store, cfg.user_id)

        # Create system message
        system_msg = _create_system_message(profile, todos, instructions)

        # Process with tool-enabled LLM
        tool_llm = llm.bind_tools([UpdateMemory], parallel_tool_calls=False)
        response = tool_llm.invoke(
            [system_msg, *state["messages"]]
        )
        return {"messages": [response]}

    return task_maistro


def _create_profile_updater(llm: ChatOpenAI):
    """Create profile update node with TrustCall extractor.

    This node uses TrustCall to extract and update user profile information
    from conversation context, maintaining structured user data over time.

    Args:
        llm: ChatOpenAI model instance for extraction

    Returns:
        Function that updates user profile based on conversation
    """
    profile_extractor = create_extractor(
        llm, tools=[Profile], tool_choice="Profile", enable_inserts=True
    )

    def update_profile(state: MessagesState, config: RunnableConfig, store: BaseStore):
        """Update user profile using TrustCall extractor.

        This function processes conversation messages to extract and update
        structured user profile information, storing the results in memory.

        Args:
            state: Current conversation state with message history
            config: Runnable configuration with user settings
            store: Memory store for persisting profile updates

        Returns:
            State with tool message confirming profile update
        """
        cfg = _create_config_helper(config)
        namespace = ("profile", cfg.user_id)

        # Prepare existing profile documents for TrustCall
        existing_docs = _prepare_existing_docs(store, namespace, "Profile")

        # Format instruction with current timestamp
        formatted_instruction = TRUSTCALL_INSTRUCTION.format(
            time=datetime.now().isoformat()
        )

        # Prepare messages for extraction
        messages = _format_trustcall_messages(
            state["messages"],
            formatted_instruction,
        )

        # Extract and update profile using TrustCall
        result = profile_extractor.invoke(
            {"messages": messages, "existing": existing_docs}
        )

        # Store extracted profile responses
        responses = result.get("responses") or []
        metadata = result.get("response_metadata") or []
        _store_trustcall_responses(store, namespace, responses, metadata)

        # Return tool response
        tool_call_id = _last_tool_call_id(state)
        return {
            "messages": [
                _tool_message("updated profile", tool_call_id),
            ]
        }

    return update_profile


def _create_todo_extractor_with_monitoring(llm: ChatOpenAI):
    """Create todo extractor with tool call monitoring.

    Returns:
        Tuple of (todo_extractor, spy) for monitoring tool calls
    """
    spy = ToolCallSpy()
    todo_extractor = create_extractor(
        llm,
        tools=[ToDo],
        tool_choice="ToDo",
        enable_inserts=True,
    ).with_listeners(on_end=spy)
    return todo_extractor, spy


def _process_todo_update(
    todo_extractor, messages, existing_docs, namespace, store
):
    """Process todo update using TrustCall extractor.

    Args:
        todo_extractor: TrustCall extractor for todos
        messages: Formatted messages for extraction
        existing_docs: Existing todo documents
        namespace: Store namespace
        store: Memory store instance

    Returns:
        Tool call spy for monitoring
    """
    result = todo_extractor.invoke(
        {"messages": messages, "existing": existing_docs}
    )

    responses = result.get("responses") or []
    metadata = result.get("response_metadata") or []
    _store_trustcall_responses(store, namespace, responses, metadata)

    return result


def _create_todos_updater(llm: ChatOpenAI):
    """Create todos update node with TrustCall extractor and tool call monitoring."""

    def update_todos(state: MessagesState, config: RunnableConfig, store: BaseStore):
        """Update user todos using TrustCall extractor with monitoring."""
        cfg = _create_config_helper(config)
        namespace = ("todo", cfg.user_id)

        # Prepare data for extraction
        existing_docs = _prepare_existing_docs(store, namespace, "ToDo")
        formatted_instruction = TRUSTCALL_INSTRUCTION.format(
            time=datetime.now().isoformat()
        )
        messages = _format_trustcall_messages(
            state["messages"], formatted_instruction
        )

        # Create extractor with monitoring
        todo_extractor, spy = _create_todo_extractor_with_monitoring(llm)

        # Process todo update
        _process_todo_update(todo_extractor, messages, existing_docs, namespace, store)

        # Return tool response
        summary = summarize_tool_calls(spy.called_tools, schema_name="ToDo")
        tool_call_id = _last_tool_call_id(state)
        return {
            "messages": [
                _tool_message(summary, tool_call_id)
            ]
        }

    return update_todos


def _create_instructions_updater(llm: ChatOpenAI):
    """Create instructions update node."""

    def update_instructions(
        state: MessagesState, config: RunnableConfig, store: BaseStore
    ):
        """Update user instructions based on conversation context."""
        cfg = _create_config_helper(config)
        namespace = ("instructions", cfg.user_id)
        existing = store.get(namespace, "user_instructions")

        prompt = CREATE_INSTRUCTIONS_PROMPT.format(
            current_instructions=existing.value.get("memory") if existing else ""
        )

        update_request = [
            SystemMessage(content=prompt),
            *state["messages"][:-1],
            HumanMessage(
                content="Please update the instructions based on the conversation."
            ),
        ]

        reflection = llm.invoke(update_request)
        store.put(namespace, "user_instructions", {"memory": reflection.content})

        tool_call_id = _last_tool_call_id(state)
        return {
            "messages": [
                _tool_message("updated instructions", tool_call_id),
            ]
        }

    return update_instructions


def _create_router():
    """Create message routing function for conditional edges."""

    def route_message(
        state: MessagesState, config: RunnableConfig, store: BaseStore
    ) -> Literal[END, "update_profile", "update_todos", "update_instructions"]:
        """Route messages to appropriate update nodes based on tool calls."""
        message = state["messages"][-1]
        if not getattr(message, "tool_calls", None):
            return END

        update_type = message.tool_calls[0]["args"]["update_type"]
        if update_type == "user":
            return "update_profile"
        if update_type == "todo":
            return "update_todos"
        if update_type == "instructions":
            return "update_instructions"
        return END

    return route_message


def _create_all_nodes(llm: ChatOpenAI):
    """Create all node functions for the multi-memory agent.

    Args:
        llm: ChatOpenAI model instance

    Returns:
        Tuple of (task_maistro, update_profile, update_todos, update_instructions, route_message)
    """
    task_maistro = _create_task_maistro_node(llm)
    update_profile = _create_profile_updater(llm)
    update_todos = _create_todos_updater(llm)
    update_instructions = _create_instructions_updater(llm)
    route_message = _create_router()
    return task_maistro, update_profile, update_todos, update_instructions, route_message


def _build_multi_memory_graph_structure(builder, nodes):
    """Build the graph structure for multi-memory agent.

    Args:
        builder: StateGraph builder instance
        nodes: Tuple of (task_maistro, update_profile, update_todos, update_instructions, route_message)
    """
    task_maistro, update_profile, update_todos, update_instructions, route_message = nodes

    builder.add_node("task_maistro", task_maistro)
    builder.add_node("update_profile", update_profile)
    builder.add_node("update_todos", update_todos)
    builder.add_node("update_instructions", update_instructions)
    builder.add_edge(START, "task_maistro")
    builder.add_conditional_edges("task_maistro", route_message)
    builder.add_edge("update_profile", "task_maistro")
    builder.add_edge("update_todos", "task_maistro")
    builder.add_edge("update_instructions", "task_maistro")


def _initialize_multi_memory_builder():
    """Initialize StateGraph builder for multi-memory agent.

    Returns:
        StateGraph builder configured for MessagesState and MemoryConfiguration
    """
    return StateGraph(MessagesState, config_schema=MemoryConfiguration)


def _compile_and_save_multi_memory_graph(builder):
    """Compile multi-memory graph and save visualization.

    Args:
        builder: StateGraph builder instance

    Returns:
        Compiled LangGraph agent
    """
    graph = builder.compile(store=InMemoryStore(), checkpointer=MemorySaver())
    save_graph_image(graph, filename="artifacts/trustcall_memory_agent.png", xray=True)
    return graph


def build_trustcall_agent(model: ChatOpenAI | None = None):
    """Compile a multi-memory LangGraph agent powered by TrustCall.

    This agent coordinates multiple memory types (profile, todos, instructions)
    using TrustCall extractors to insert or patch structured documents based on
    conversation turns.

    Args:
        model: Optional ChatOpenAI model instance. If None, creates default LLM.

    Returns:
        Compiled LangGraph agent with memory management capabilities.
    """
    llm = model or create_llm()

    # Create all node functions
    nodes = _create_all_nodes(llm)

    # Build the graph
    builder = _initialize_multi_memory_builder()
    _build_multi_memory_graph_structure(builder, nodes)

    # Compile and return the graph
    return _compile_and_save_multi_memory_graph(builder)


def main() -> None:
    require_llm_provider_api_key()
    maybe_enable_langsmith()
    graph = build_trustcall_agent()
    config = {"configurable": {"thread_id": "memory-agent-demo", "user_id": "leslie"}}

    script = [
        "My name is Leslie. I live in San Francisco with my wife and baby daughter.",
        "Please remember that I like gathering restaurant suggestions for the weekend.",
        "Add a task to book swim lessons for the baby.",
        "When you add tasks, include specific local service providers.",
        "I need to fix the jammed Yale lock on the door by the end of the month.",
    ]

    for text in script:
        for chunk in graph.stream(
            {"messages": [HumanMessage(content=text)]}, config, stream_mode="values"
        ):
            pretty_print_messages(
                chunk["messages"][-1:], header=f"Assistant reply to: {text}"
            )

    # Demonstrate a follow-up conversation using accumulated memory.
    follow_up = [
        HumanMessage(
            content="I have 30 minutes before lunch. What task should I tackle?"
        )
    ]
    for chunk in graph.stream({"messages": follow_up}, config, stream_mode="values"):
        pretty_print_messages(chunk["messages"][-1:], header="Follow-up conversation")


if __name__ == "__main__":
    main()
