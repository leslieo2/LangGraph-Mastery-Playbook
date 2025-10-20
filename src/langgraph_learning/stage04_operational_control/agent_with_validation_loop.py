"""
Validation Loop: Repeated Interrupts for Clean Input

=== PROBLEM STATEMENT ===
Agents that rely on human-provided data often receive malformed input. Without a safe
retry mechanism, invalid responses either crash the node or force users to restart the
entire run from scratch.

=== CORE SOLUTION ===
This lesson keeps the run alive by looping on `interrupt()`. Each resume value is
validated; bad data updates the prompt and triggers another interrupt, while good data
lets the node finish and persist the cleaned value.

=== KEY INNOVATION ===
- **Prompt Recycling** – Resume values flow back into the same node for validation.
- **Friendly Re-Prompts** – The interrupt message adapts using prior input as context.
- **Persistent Cursor** – Memory-backed threads let operators retry indefinitely.

=== COMPARISON WITH DYNAMIC GUARDS ===
| agent_with_dynamic_interruption | This lesson (validation loop) |
|---------------------------------|--------------------------------|
| Interrupts when state hits a guardrail | Interrupts until human supplies valid data |
| Requires external state patching       | Repairs data in-node via guided prompts   |
| Prints raw state snapshots             | Returns just the cleaned value            |

What You'll Learn
1. Build a node that loops on `interrupt()` without losing progress.
2. Resume runs with `Command(resume=...)` for both invalid and valid responses.
3. Store the finalized value in state once validation passes.

Lesson Flow
1. Compile a simple form-collection graph with checkpointing enabled.
2. Invoke the graph to capture the initial interrupt prompt.
3. Resume once with invalid data, then again with a valid value to complete the run.
"""

from __future__ import annotations

from typing import TypedDict
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from src.langgraph_learning.utils import save_graph_image


class FormState(TypedDict, total=False):
    """State for collecting a single numeric field from a human."""

    age: int


def _assemble_validation_graph(memory: MemorySaver):
    """Return validation graph compiled with supplied checkpointer."""

    def collect_age(state: FormState):
        prompt = "What is your age?"

        while True:
            answer = interrupt({"prompt": prompt})
            if isinstance(answer, int) and answer > 0:
                return {"age": answer}

            prompt = f"'{answer}' is not a valid age. Please enter a positive integer."

    builder = StateGraph(FormState)
    builder.add_node("collect_age", collect_age)
    builder.add_edge(START, "collect_age")
    builder.add_edge("collect_age", END)

    graph = builder.compile(checkpointer=memory)
    return graph


def build_validation_graph():
    """Create a graph that keeps asking for age until it receives a positive integer."""
    graph = _assemble_validation_graph(MemorySaver())
    save_graph_image(graph, filename="artifacts/agent_with_validation_loop.png")
    return graph


def _thread(label: str) -> dict:
    return {"configurable": {"thread_id": f"{label}-{uuid4().hex}"}}


def prompt_for_age(graph):
    """Invoke the graph once to capture the first interrupt payload."""

    config = _thread("age-form")
    initial = graph.invoke({}, config=config)
    interrupt_payloads = initial.get("__interrupt__", [])
    if interrupt_payloads:
        print("Initial prompt:", interrupt_payloads[0].value)
    else:
        print("No interrupt encountered; validation loop may be misconfigured.")
    return config, interrupt_payloads


def retry_with_invalid_input(graph, config: dict):
    """Resume the run with bad data to show the updated interrupt message."""

    result = graph.invoke(Command(resume="twenty"), config=config)
    interrupt_payloads = result.get("__interrupt__", [])
    if interrupt_payloads:
        print("Retry prompt:", interrupt_payloads[0].value)
    return interrupt_payloads


def supply_valid_age(graph, config: dict):
    """Provide a valid age so the node completes and the run returns clean state."""

    final_state = graph.invoke(Command(resume=27), config=config)
    print("Collected age:", final_state.get("age"))


def main() -> None:
    graph = build_validation_graph()

    config, interrupts = prompt_for_age(graph)
    if not interrupts:
        return

    retry_interrupts = retry_with_invalid_input(graph, config)
    if not retry_interrupts:
        return

    supply_valid_age(graph, config)


if __name__ == "__main__":
    main()


def studio_graph(config: RunnableConfig | None = None):
    """Studio entry point for the validation loop demo."""
    return _assemble_validation_graph(MemorySaver())
