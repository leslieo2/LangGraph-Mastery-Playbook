"""What You'll Learn
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

from pydantic import BaseModel, Field
from trustcall import create_extractor

from langchain_core.messages import HumanMessage, SystemMessage, merge_message_runs
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
    ToolCallSpy,
    maybe_enable_langsmith,
    pretty_print_messages,
    require_env,
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


def build_trustcall_agent(model: ChatOpenAI | None = None):
    """Compile a multi-memory LangGraph agent powered by TrustCall."""
    llm = model or create_llm()
    profile_extractor = create_extractor(
        llm, tools=[Profile], tool_choice="Profile", enable_inserts=True
    )

    def _cfg(config: RunnableConfig) -> MemoryConfiguration:
        return MemoryConfiguration.from_runnable_config(config)

    def _load_profile(store: BaseStore, user_id: str) -> dict | None:
        existing = store.search(("profile", user_id))
        return existing[0].value if existing else None

    def _load_todos(store: BaseStore, user_id: str) -> list[dict]:
        return [memory.value for memory in store.search(("todo", user_id))]

    def _load_instructions(store: BaseStore, user_id: str) -> str:
        existing = store.get(("instructions", user_id), "user_instructions")
        return existing.value.get("memory", "") if existing else ""

    def task_maistro(state: MessagesState, config: RunnableConfig, store: BaseStore):
        cfg = _cfg(config)
        profile = _load_profile(store, cfg.user_id) or {}
        todos = _load_todos(store, cfg.user_id)
        instructions = _load_instructions(store, cfg.user_id)
        todo_text = "\n".join(map(str, todos)) or "No tasks saved."
        system_msg = SYSTEM_PROMPT.format(
            user_profile=profile or "No profile stored.",
            todo=todo_text,
            instructions=instructions or "No custom instructions.",
        )
        tool_llm = llm.bind_tools([UpdateMemory], parallel_tool_calls=False)
        response = tool_llm.invoke(
            [SystemMessage(content=system_msg), *state["messages"]]
        )
        return {"messages": [response]}

    def update_profile(state: MessagesState, config: RunnableConfig, store: BaseStore):
        cfg = _cfg(config)
        namespace = ("profile", cfg.user_id)
        existing = store.search(namespace)
        existing_docs = (
            [(item.key, "Profile", item.value) for item in existing]
            if existing
            else None
        )
        formatted_instruction = TRUSTCALL_INSTRUCTION.format(
            time=datetime.now().isoformat()
        )
        messages = list(
            merge_message_runs(
                [SystemMessage(content=formatted_instruction), *state["messages"][:-1]]
            )
        )
        result = profile_extractor.invoke(
            {"messages": messages, "existing": existing_docs}
        )
        for response, meta in zip(result["responses"], result["response_metadata"]):
            key = meta.get("json_doc_id") or str(uuid.uuid4())
            store.put(namespace, key, response.model_dump(mode="json"))
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        return {
            "messages": [
                {
                    "role": "tool",
                    "content": "updated profile",
                    "tool_call_id": tool_call_id,
                }
            ]
        }

    def update_todos(state: MessagesState, config: RunnableConfig, store: BaseStore):
        cfg = _cfg(config)
        namespace = ("todo", cfg.user_id)
        existing = store.search(namespace)
        existing_docs = (
            [(item.key, "ToDo", item.value) for item in existing] if existing else None
        )
        formatted_instruction = TRUSTCALL_INSTRUCTION.format(
            time=datetime.now().isoformat()
        )
        messages = list(
            merge_message_runs(
                [SystemMessage(content=formatted_instruction), *state["messages"][:-1]]
            )
        )
        spy = ToolCallSpy()
        todo_extractor = create_extractor(
            llm,
            tools=[ToDo],
            tool_choice="ToDo",
            enable_inserts=True,
        ).with_listeners(on_end=spy)
        result = todo_extractor.invoke(
            {"messages": messages, "existing": existing_docs}
        )
        for response, meta in zip(result["responses"], result["response_metadata"]):
            key = meta.get("json_doc_id") or str(uuid.uuid4())
            store.put(namespace, key, response.model_dump(mode="json"))
        summary = summarize_tool_calls(spy.called_tools, schema_name="ToDo")
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        return {
            "messages": [
                {"role": "tool", "content": summary, "tool_call_id": tool_call_id}
            ]
        }

    def update_instructions(
        state: MessagesState, config: RunnableConfig, store: BaseStore
    ):
        cfg = _cfg(config)
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
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        return {
            "messages": [
                {
                    "role": "tool",
                    "content": "updated instructions",
                    "tool_call_id": tool_call_id,
                }
            ]
        }

    def route_message(
        state: MessagesState, config: RunnableConfig, store: BaseStore
    ) -> Literal[END, "update_profile", "update_todos", "update_instructions"]:
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

    builder = StateGraph(MessagesState, config_schema=MemoryConfiguration)
    builder.add_node("task_maistro", task_maistro)
    builder.add_node("update_profile", update_profile)
    builder.add_node("update_todos", update_todos)
    builder.add_node("update_instructions", update_instructions)
    builder.add_edge(START, "task_maistro")
    builder.add_conditional_edges("task_maistro", route_message)
    builder.add_edge("update_profile", "task_maistro")
    builder.add_edge("update_todos", "task_maistro")
    builder.add_edge("update_instructions", "task_maistro")

    graph = builder.compile(store=InMemoryStore(), checkpointer=MemorySaver())
    save_graph_image(graph, filename="artifacts/trustcall_memory_agent.png", xray=True)
    return graph


def main() -> None:
    require_env("OPENAI_API_KEY")
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
