"""
Advanced Memory System: Multi-Memory Coordination with Intelligent Routing

=== PROBLEM STATEMENT ===
Real-world applications often need to manage multiple types of user information simultaneously:
- User profiles (name, location, interests)
- Task lists and todos
- Custom instructions and preferences

How can we build an agent that intelligently decides which memory type to update based on conversation context, rather than updating everything every time?

=== CORE SOLUTION ===
This system introduces intelligent routing between multiple memory types using a decision-making node that analyzes conversation context to determine which memory bucket needs updating.

=== KEY INNOVATION ===
- **Intelligent Routing**: Task Maestro node decides which memory type to update
- **Multi-Memory Coordination**: Profiles, todos, and instructions managed in parallel
- **Dynamic Decision Making**: UpdateMemory TypedDict enables context-aware routing
- **Selective Updates**: Only relevant memories are updated, improving efficiency

=== COMPARISON WITH OTHER MEMORY SYSTEMS ===
| Single Memory Systems | Multi-Memory Coordination (this file) |
|----------------------|--------------------------------------|
| Updates single memory type | Routes between multiple memory types |
| Fixed update pattern | Context-aware decision making |
| Simpler architecture | More sophisticated memory management |

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
from typing import Literal, Optional, TypedDict

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field
from trustcall import create_extractor

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
        min_length=1,
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


def _create_task_orchestra_node(llm: ChatOpenAI):
    """Create the task orchestra node for processing user requests.

    CORE TEACHING CONCEPT: This node demonstrates intelligent routing between
    multiple memory types based on conversation context.
    """

    def task_orchestra(state: MessagesState, config: RunnableConfig, store: BaseStore):
        """Process user request and decide which memory to update.

        KEY INNOVATION: Instead of updating all memory types every time,
        this node analyzes conversation context to determine which specific
        memory bucket needs updating.
        """
        cfg = MemoryConfiguration.from_runnable_config(config)

        # Load current memory state for context
        profile = store.search(("profile", cfg.user_id))
        todos = store.search(("todo", cfg.user_id))
        instructions = store.get(("instructions", cfg.user_id), "user_instructions")

        # Format memory for system prompt
        profile_text = profile[0].value if profile else "No profile stored."
        todo_text = "\n".join(str(item.value) for item in todos) or "No tasks saved."
        instructions_text = (
            instructions.value.get("memory", "")
            if instructions
            else "No custom instructions."
        )

        # Create system message with memory context
        system_content = SYSTEM_PROMPT.format(
            user_profile=profile_text,
            todo=todo_text,
            instructions=instructions_text,
        )
        system_msg = SystemMessage(content=system_content)

        # Process with tool-enabled LLM for routing decision
        tool_llm = llm.bind_tools([UpdateMemory], parallel_tool_calls=False)
        response = tool_llm.invoke([system_msg, *state["messages"]])
        return {"messages": [response]}

    return task_orchestra


def _create_profile_updater(llm: ChatOpenAI):
    """Create profile update node with TrustCall extractor.

    CORE TEACHING CONCEPT: Demonstrates TrustCall's incremental update mechanism
    for structured user profiles using JSON Patch.
    """
    profile_extractor = create_extractor(
        llm, tools=[Profile], tool_choice="Profile", enable_inserts=True
    )

    def update_profile(state: MessagesState, config: RunnableConfig, store: BaseStore):
        """Update user profile using TrustCall extractor.

        KEY INNOVATION: TrustCall uses JSON Patch to update only changed fields,
        making it more efficient than regenerating the entire profile.
        """
        cfg = MemoryConfiguration.from_runnable_config(config)
        namespace = ("profile", cfg.user_id)

        # Load existing profile for incremental updates
        existing_items = store.search(namespace)
        existing_docs = (
            [(item.key, "Profile", item.value) for item in existing_items]
            if existing_items
            else None
        )

        # Prepare messages for TrustCall extraction
        instruction = TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
        messages = [SystemMessage(content=instruction), *state["messages"][:-1]]

        # Extract and update profile using TrustCall
        result = profile_extractor.invoke(
            {"messages": messages, "existing": existing_docs}
        )

        # Store updated profile
        for response, meta in zip(result["responses"], result["response_metadata"]):
            key = meta.get("json_doc_id") or str(uuid.uuid4())
            store.put(namespace, key, response.model_dump(mode="json"))

        # Return confirmation
        message = state["messages"][-1]
        tool_call_id = (
            message.tool_calls[0]["id"]
            if getattr(message, "tool_calls", None)
            else None
        )
        return {
            "messages": [
                {
                    "role": "tool",
                    "content": "updated profile",
                    "tool_call_id": tool_call_id,
                }
            ]
        }

    return update_profile


def _create_todos_updater(llm: ChatOpenAI):
    """Create todos update node with TrustCall extractor.

    CORE TEACHING CONCEPT: Shows how TrustCall can manage collections of
    structured tasks with incremental updates.
    """
    todo_extractor = create_extractor(
        llm, tools=[ToDo], tool_choice="ToDo", enable_inserts=True
    )

    def update_todos(state: MessagesState, config: RunnableConfig, store: BaseStore):
        """Update user todos using TrustCall extractor.

        KEY INNOVATION: TrustCall enables both inserting new todos and
        updating existing ones in a single operation.
        """
        cfg = MemoryConfiguration.from_runnable_config(config)
        namespace = ("todo", cfg.user_id)

        # Load existing todos
        existing_items = store.search(namespace)
        existing_docs = (
            [(item.key, "ToDo", item.value) for item in existing_items]
            if existing_items
            else None
        )

        # Prepare messages for extraction
        instruction = TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
        messages = [SystemMessage(content=instruction), *state["messages"][:-1]]

        # Extract and update todos
        result = todo_extractor.invoke(
            {"messages": messages, "existing": existing_docs}
        )

        # Store updated todos
        for response, meta in zip(result["responses"], result["response_metadata"]):
            key = meta.get("json_doc_id") or str(uuid.uuid4())
            store.put(namespace, key, response.model_dump(mode="json"))

        # Return confirmation
        message = state["messages"][-1]
        tool_call_id = (
            message.tool_calls[0]["id"]
            if getattr(message, "tool_calls", None)
            else None
        )
        return {
            "messages": [
                {
                    "role": "tool",
                    "content": "updated todos",
                    "tool_call_id": tool_call_id,
                }
            ]
        }

    return update_todos


def _create_instructions_updater(llm: ChatOpenAI):
    """Create instructions update node.

    CORE TEACHING CONCEPT: Shows how to update unstructured text-based
    instructions that guide the agent's behavior.
    """

    def update_instructions(
        state: MessagesState, config: RunnableConfig, store: BaseStore
    ):
        """Update user instructions based on conversation context.

        KEY INNOVATION: Demonstrates a different approach to memory updates
        that doesn't use TrustCall, showing the flexibility of the architecture.
        """
        cfg = MemoryConfiguration.from_runnable_config(config)
        namespace = ("instructions", cfg.user_id)

        # Load current instructions
        existing = store.get(namespace, "user_instructions")
        current_instructions = existing.value.get("memory", "") if existing else ""

        # Create update prompt
        prompt = CREATE_INSTRUCTIONS_PROMPT.format(
            current_instructions=current_instructions
        )

        # Generate updated instructions
        update_request = [
            SystemMessage(content=prompt),
            *state["messages"][:-1],
            HumanMessage(
                content="Please update the instructions based on the conversation."
            ),
        ]

        reflection = llm.invoke(update_request)
        store.put(namespace, "user_instructions", {"memory": reflection.content})

        # Return confirmation
        message = state["messages"][-1]
        tool_call_id = (
            message.tool_calls[0]["id"]
            if getattr(message, "tool_calls", None)
            else None
        )
        return {
            "messages": [
                {
                    "role": "tool",
                    "content": "updated instructions",
                    "tool_call_id": tool_call_id,
                }
            ]
        }

    return update_instructions


def _create_router():
    """Create message routing function for conditional edges.

    CORE TEACHING CONCEPT: Demonstrates how LangGraph's conditional edges
    enable dynamic routing based on conversation context.
    """

    def route_message(
        state: MessagesState, config: RunnableConfig, store: BaseStore
    ) -> Literal[END, "update_profile", "update_todos", "update_instructions"]:
        """Route messages to appropriate update nodes based on tool calls.

        KEY INNOVATION: This routing logic enables selective memory updates
        instead of updating all memory types every time.
        """
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


def build_trustcall_agent(model: ChatOpenAI | None = None):
    """Compile a multi-memory LangGraph agent powered by TrustCall.

    This agent coordinates multiple memory types (profile, todos, instructions)
    using TrustCall extractors to insert or patch structured documents based on
    conversation turns.

    GRAPH FLOW:
    START → task_orchestra → [conditional routing] → update_memory → task_orchestra

    KEY INNOVATION: Intelligent routing enables selective memory updates
    based on conversation context, improving efficiency.

    Args:
        model: Optional ChatOpenAI model instance. If None, creates default LLM.

    Returns:
        Compiled LangGraph agent with memory management capabilities.
    """
    llm = model or create_llm()

    # Create all node functions
    task_orchestra = _create_task_orchestra_node(llm)
    update_profile = _create_profile_updater(llm)
    update_todos = _create_todos_updater(llm)
    update_instructions = _create_instructions_updater(llm)
    route_message = _create_router()

    # Build the graph
    builder = StateGraph(MessagesState, config_schema=MemoryConfiguration)
    builder.add_node("task_orchestra", task_orchestra)
    builder.add_node("update_profile", update_profile)
    builder.add_node("update_todos", update_todos)
    builder.add_node("update_instructions", update_instructions)
    builder.add_edge(START, "task_orchestra")
    builder.add_conditional_edges("task_orchestra", route_message)
    builder.add_edge("update_profile", "task_orchestra")
    builder.add_edge("update_todos", "task_orchestra")
    builder.add_edge("update_instructions", "task_orchestra")

    # Compile and return the graph
    graph = builder.compile(store=InMemoryStore(), checkpointer=MemorySaver())
    save_graph_image(
        graph, filename="artifacts/agent_with_multi_memory_coordination.png", xray=True
    )
    return graph


def main() -> None:
    require_llm_provider_api_key()
    maybe_enable_langsmith()
    graph = build_trustcall_agent()
    config = {"configurable": {"thread_id": "memory-agent-demo", "user_id": "leslie"}}

    print("=== DEMONSTRATION: Multi-Memory Coordination with Intelligent Routing ===\n")

    script = [
        "My name is Leslie. I live in Shanghai",
        "Please remember that I like gathering restaurant suggestions for the weekend.",
        "Add a task to book swim lessons.",
        "When you add tasks, include specific local service providers.",
        "I need to fix the jammed Yale lock on the door by the end of the month.",
    ]

    for text in script:
        print(f"User: '{text}'")
        for chunk in graph.stream(
            {"messages": [HumanMessage(content=text)]}, config, stream_mode="values"
        ):
            pretty_print_messages(
                chunk["messages"][-1:], header=f"Assistant reply to: {text}"
            )

    # Demonstrate a follow-up conversation using accumulated memory.
    print("\n=== FOLLOW-UP: Using Accumulated Memory ===")
    follow_up = [
        HumanMessage(
            content="I have 30 minutes before lunch. What task should I tackle?"
        )
    ]
    for chunk in graph.stream({"messages": follow_up}, config, stream_mode="values"):
        pretty_print_messages(chunk["messages"][-1:], header="Follow-up conversation")

    print("\n=== KEY TAKEAWAY ===")
    print(
        "Intelligent routing enables selective memory updates based on conversation context."
    )
    print("This is more efficient than updating all memory types every time.")


if __name__ == "__main__":
    main()
