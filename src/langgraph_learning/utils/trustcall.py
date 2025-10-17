"""TrustCall utility functions for monitoring and debugging tool calls.

This module provides utilities for working with TrustCall extractors, including:
- ToolCallSpy: A listener that captures tool call payloads for debugging
- ToolChange: A dataclass representing tool call changes
- summarize_tool_calls: Function to pretty-print tool call changes
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


class ToolCallSpy:
    """Collect tool-call payloads emitted by TrustCall for debugging.

    This class can be used as a listener to monitor tool calls made by
    TrustCall extractors during execution. It captures the tool call
    payloads for later analysis and debugging.

    Attributes:
        called_tools: List of tool call payloads captured during execution
    """

    def __init__(self) -> None:
        """Initialize an empty tool call spy."""
        self.called_tools: list[list[dict[str, Any]] | None] = []

    def __call__(self, run: Any) -> None:
        """Process a run to capture tool calls.

        This method is called by the LangChain/LangGraph framework when
        used as a listener. It traverses the run tree to find chat model
        runs and captures their tool calls.

        Args:
            run: The run object to process for tool calls
        """
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
    """Represents a single tool call change from TrustCall.

    Attributes:
        kind: Type of change - "new" for new documents, "update" for updates
        payload: Dictionary containing the change details
    """

    kind: str  # "new" or "update"
    payload: dict[str, Any]


def summarize_tool_calls(
    tool_calls: Iterable[Iterable[dict[str, Any]]], schema_name: str = "Memory"
) -> str:
    """Pretty-print the changes represented by TrustCall tool calls.

    This function processes tool calls captured by ToolCallSpy and creates
    a human-readable summary of the changes made by TrustCall extractors.

    Args:
        tool_calls: Iterable of tool call groups from ToolCallSpy
        schema_name: Name of the schema being processed (e.g., "Memory", "Profile")

    Returns:
        Formatted string summarizing the tool call changes
    """
    changes: list[ToolChange] = []

    # Process all tool call groups
    for group in tool_calls:
        for call in group:
            name = call.get("name")
            args = call.get("args", {})

            if name == "PatchDoc":
                # Handle document updates
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
                # Handle new document creation
                changes.append(ToolChange(kind="new", payload=args))

    if not changes:
        return "No tool changes recorded."

    # Format the changes for display
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
