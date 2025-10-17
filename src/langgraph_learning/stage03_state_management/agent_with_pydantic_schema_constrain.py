"""This LangGraph schema lab runs the same branching workflow across `TypedDict`, dataclass, and Pydantic states to compare validation behavior.

What You'll Learn
1. Compare different state schema strategies: `TypedDict`, dataclass, and Pydantic models.
2. See how LangGraph validates input and output when you swap schema types.
3. Understand validation failures by intentionally triggering Pydantic errors.

Lesson Flow
1. Define equivalent state representations using three schema paradigms.
2. Build a reusable `build_graph` helper and run it with typed dict and dataclass states.
3. Demonstrate Pydantic validation, visualize each graph, and print invocation results.
"""

import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, TypedDict, Type

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ValidationError, field_validator

from src.langgraph_learning.utils import save_graph_image


class TypedDictState(TypedDict):
    name: str
    mood: Literal["happy", "sad"]


@dataclass
class DataclassState:
    name: str
    mood: Literal["happy", "sad"]


class PydanticState(BaseModel):
    name: str
    mood: str  # "happy" or "sad"

    @field_validator("mood")
    @classmethod
    def validate_mood(cls, value: str) -> str:
        if value not in {"happy", "sad"}:
            raise ValueError("Each mood must be either 'happy' or 'sad'")
        return value


def node_2(_: Any) -> Dict[str, str]:
    print("---Node 2---")
    return {"mood": "happy"}


def node_3(_: Any) -> Dict[str, str]:
    print("---Node 3---")
    return {"mood": "sad"}


def decide_mood(_: Any) -> Literal["node_2", "node_3"]:
    return "node_2" if random.random() < 0.5 else "node_3"


def build_graph(
    state_type: Type[Any],
    node_1_fn: Callable[[Any], Dict[str, str]],
):
    builder = StateGraph(state_type)
    builder.add_node("node_1", node_1_fn)
    builder.add_node("node_2", node_2)
    builder.add_node("node_3", node_3)
    builder.add_edge(START, "node_1")
    builder.add_conditional_edges("node_1", decide_mood)
    builder.add_edge("node_2", END)
    builder.add_edge("node_3", END)
    return builder.compile()


def node_1_typed_dict(state: TypedDictState) -> Dict[str, str]:
    print("---Node 1 (TypedDict)---")
    return {"name": state["name"] + " is ... "}


def run_typed_dict_example() -> None:
    graph = build_graph(TypedDictState, node_1_typed_dict)
    print("Result:", graph.invoke({"name": "Leslie"}))


def node_1_dataclass(state: DataclassState) -> Dict[str, str]:
    print("---Node 1 (Dataclass)---")
    return {"name": state.name + " is ... "}


def run_dataclass_example() -> None:
    graph = build_graph(DataclassState, node_1_dataclass)
    # Provide a valid initial mood as a placeholder; the graph will overwrite it.
    print("Result:", graph.invoke(DataclassState(name="Leslie", mood="sad")))


def node_1_pydantic(state: PydanticState) -> Dict[str, str]:
    print("---Node 1 (Pydantic)---")
    return {"name": state.name + " is ... "}


def demonstrate_pydantic_validation() -> None:
    try:
        PydanticState(name="John Doe", mood="mad")
    except ValidationError as exc:
        print("Validation Error:", exc)


def run_pydantic_example() -> None:
    graph = build_graph(PydanticState, node_1_pydantic)
    save_graph_image(graph, filename="artifacts/agent_with_pydantic_schema_constrain.png")
    print("Result:", graph.invoke(PydanticState(name="Leslie", mood="sad")))


def main() -> None:
    run_typed_dict_example()
    run_dataclass_example()
    demonstrate_pydantic_validation()
    run_pydantic_example()


if __name__ == "__main__":
    main()
