"""
Parallel Nodes: Map-Reduce Joke Generator

=== PROBLEM STATEMENT ===
Sequential prompting can bottleneck LLM workflows: exploring alternatives one at a time
delays feedback and makes it hard to pick the best result when work could be done in
parallel.

=== CORE SOLUTION ===
This lesson builds a map-reduce LangGraph app that fans out joke generation across
subjects, collects each punchline, then chooses a winner using structured outputs for
deterministic selection.

=== KEY INNOVATION ===
- **Dynamic Fan-Out with `Send`**: Launch nodes per subject without predefining edges.
- **Reducer Semantics**: Append jokes via list reducers as parallel nodes report in.
- **Structured Selection**: Use `with_structured_output` to pick the best joke safely.

=== COMPARISON WITH SEQUENTIAL GENERATION ===
| Sequential Pipeline | Parallel Map-Reduce (this file) |
|---------------------|----------------------------------|
| Generates one joke at a time | Spawns per-subject joke generators concurrently |
| Hard to compare candidates | Central reducer aggregates and scores jokes |
| Manual selection via parsing | Structured schema ensures valid index selection |

What You'll Learn
1. Fan out work with `Send` to parallelize LangGraph nodes against dynamic inputs.
2. Accumulate state with list reducers and apply `llm.with_structured_output` for structured responses.
3. Visualize the compiled graph and stream a demo run that highlights the map and reduce phases.

Lesson Flow
1. Prepare typed state plus Pydantic helpers for subjects, jokes, and the winning punchline.
2. Use `llm.with_structured_output` to power each node, then build a map-reduce graph that generates jokes per subject and collapses them into a final choice.
3. Render the graph diagram and stream an example topic to observe each node's contribution.
"""

from __future__ import annotations

import operator
from typing import Annotated

from pydantic import BaseModel
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src.langgraph_learning.utils import (
    create_llm,
    maybe_enable_langsmith,
    require_llm_provider_api_key,
    save_graph_image,
)

SUBJECTS_PROMPT = "Generate a list of 3 sub-topics that are all related to this overall topic: {topic}."
JOKE_PROMPT = "Generate a joke about {subject}."
BEST_JOKE_PROMPT = (
    "Below are jokes about {topic}. "
    "Select the best one and return the ID of the winner, starting at 0.\n\n{jokes}"
)


class Subjects(BaseModel):
    subjects: list[str]


class Joke(BaseModel):
    joke: str


class BestJoke(BaseModel):
    id: int


class OverallState(TypedDict):
    topic: str
    subjects: list[str]
    jokes: Annotated[list[str], operator.add]
    best_selected_joke: str


class JokeState(TypedDict):
    subject: str


def build_map_reduce_app(model: str | None = None):
    """Compile the joke map-reduce LangGraph app."""
    llm = create_llm(model=model, temperature=0)
    subjects_chain = llm.with_structured_output(Subjects)
    joke_chain = llm.with_structured_output(Joke)
    best_chain = llm.with_structured_output(BestJoke)

    def generate_topics(state: OverallState):
        prompt = SUBJECTS_PROMPT.format(topic=state["topic"])
        response = subjects_chain.invoke(prompt)
        return {"subjects": response.subjects}

    def continue_to_jokes(state: OverallState):
        return [
            Send("generate_joke", {"subject": subject}) for subject in state["subjects"]
        ]

    def generate_joke(state: JokeState):
        prompt = JOKE_PROMPT.format(subject=state["subject"])
        response = joke_chain.invoke(prompt)
        return {"jokes": [response.joke]}

    def best_joke(state: OverallState):
        jokes = "\n\n".join(
            f"[{idx}] {joke}" for idx, joke in enumerate(state["jokes"])
        )
        prompt = BEST_JOKE_PROMPT.format(topic=state["topic"], jokes=jokes)
        response = best_chain.invoke(prompt)
        try:
            winner = state["jokes"][response.id]
        except (
            IndexError
        ) as exc:  # Defensive guard if the LLM returns an invalid index.
            raise ValueError(
                f"Model selected joke index {response.id}, but only {len(state['jokes'])} jokes exist."
            ) from exc
        return {"best_selected_joke": winner}

    builder = StateGraph(OverallState)
    builder.add_node("generate_topics", generate_topics)
    builder.add_node("generate_joke", generate_joke)
    builder.add_node("best_joke", best_joke)
    builder.add_edge(START, "generate_topics")
    builder.add_conditional_edges(
        "generate_topics", continue_to_jokes, ["generate_joke"]
    )
    builder.add_edge("generate_joke", "best_joke")
    builder.add_edge("best_joke", END)
    return builder.compile()


def stream_demo(app, topic: str = "animals") -> None:
    """Stream the map-reduce execution for the provided topic."""
    print(f"\n=== Map-Reduce Demo: {topic} ===")
    final_update: dict[str, dict] | None = None
    for update in app.stream({"topic": topic}):
        print(update)
        final_update = update
    if final_update and "best_joke" in final_update:
        print("\nWinning joke:", final_update["best_joke"]["best_selected_joke"])


def main() -> None:
    require_llm_provider_api_key()
    maybe_enable_langsmith(project="langgraph-map-reduce")
    app = build_map_reduce_app()
    save_graph_image(app, filename="artifacts/agent_with_parallel_nodes.png")
    stream_demo(app)


if __name__ == "__main__":
    main()
