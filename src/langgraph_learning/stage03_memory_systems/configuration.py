"""Shared configuration helpers for Stage 03 memory tutorials."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig


@dataclass(kw_only=True)
class MemoryConfiguration:
    """Runtime configuration for memory-oriented LangGraph demos."""

    user_id: str = "default-user"

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "MemoryConfiguration":
        """Create an instance from a RunnableConfig."""
        configurable: dict[str, Any] = {}
        if config and "configurable" in config:
            value = config["configurable"]
            if isinstance(value, dict):
                configurable = value

        init_values: dict[str, Any] = {}
        for field in fields(cls):
            if not field.init:
                continue
            env_value = os.environ.get(field.name.upper())
            if env_value:
                init_values[field.name] = env_value
                continue
            config_value = configurable.get(field.name)
            if config_value:
                init_values[field.name] = config_value

        return cls(**init_values)
