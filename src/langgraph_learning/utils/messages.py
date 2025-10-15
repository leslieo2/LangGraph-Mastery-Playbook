from __future__ import annotations

from typing import Iterable

from langchain_core.messages import BaseMessage


def pretty_print_messages(
    messages: Iterable[BaseMessage], *, header: str | None = None
) -> None:
    """Print LangChain messages in a readable format."""
    if header:
        print(f"\n{header}")

    for message in messages:
        message.pretty_print()
