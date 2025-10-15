from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


class ToolCallSpy:
    """Collect tool-call payloads emitted by TrustCall for debugging."""

    def __init__(self) -> None:
        self.called_tools: list[list[dict[str, Any]] | None] = []

    def __call__(self, run: Any) -> None:
        queue = [run]
        while queue:
            current = queue.pop()
            for child in getattr(current, "child_runs", []) or []:
                queue.append(child)
            if getattr(current, "run_type", None) == "chat_model":
                generations = current.outputs.get("generations", [])
                if generations and generations[0]:
                    tool_calls = generations[0][0]["message"]["kwargs"].get(
                        "tool_calls"
                    )
                    self.called_tools.append(tool_calls)


@dataclass
class ToolChange:
    kind: str  # "new" or "update"
    payload: dict[str, Any]


def summarize_tool_calls(
    tool_calls: Iterable[Iterable[dict[str, Any]]], schema_name: str = "Memory"
) -> str:
    """Pretty-print the changes represented by TrustCall tool calls."""
    changes: list[ToolChange] = []
    for group in tool_calls:
        for call in group:
            name = call.get("name")
            args = call.get("args", {})
            if name == "PatchDoc":
                changes.append(
                    ToolChange(
                        kind="update",
                        payload={
                            "doc_id": args.get("json_doc_id"),
                            "planned_edits": args.get("planned_edits"),
                            "value": args.get("patches", [{}])[0].get("value"),
                        },
                    )
                )
            elif name == schema_name:
                changes.append(ToolChange(kind="new", payload=args))

    if not changes:
        return "No tool changes recorded."

    lines: list[str] = []
    for change in changes:
        if change.kind == "update":
            details = change.payload
            lines.append(
                (
                    f"Document {details.get('doc_id')} updated\n"
                    f"  Planned edits: {details.get('planned_edits')}\n"
                    f"  New value: {details.get('value')}"
                ).rstrip()
            )
        else:
            lines.append(f"New {schema_name} created: {change.payload}")

    return "\n\n".join(lines)
