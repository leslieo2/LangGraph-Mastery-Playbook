from __future__ import annotations

import os


def require_env(var_name: str, *, message: str | None = None) -> None:
    """Ensure an environment variable is present before running a demo."""
    if not os.environ.get(var_name):
        detail = message or f"{var_name} must be set."
        raise RuntimeError(detail)


def maybe_enable_langsmith(project: str = "langchain-academy") -> None:
    """Enable LangSmith tracing when credentials are available."""
    if os.environ.get("LANGSMITH_API_KEY"):
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", project)
