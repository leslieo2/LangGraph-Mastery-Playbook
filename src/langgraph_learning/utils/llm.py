from __future__ import annotations

import os
from typing import Any

from langchain_openai import ChatOpenAI

DEFAULT_LLM_MODEL = "gpt-5-nano"
DEFAULT_LLM_TEMPERATURE = 0
DEFAULT_LLM_PROVIDER = "openai"

_PROVIDER_SETTINGS = {
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "default_base_url": None,
    },
    "openrouter": {
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url_env": "OPENROUTER_BASE_URL",
        "default_base_url": "https://openrouter.ai/api/v1",
    },
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "default_base_url": "https://api.deepseek.com/v1",
    },
}


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value:
            return value
    return None


def _from_env(var_name: str | None) -> str | None:
    return os.getenv(var_name) if var_name else None


def _coerce_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def create_llm(
    *,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    **overrides: Any,
) -> ChatOpenAI:
    """Return a ChatOpenAI instance using shared defaults and provider-specific config."""
    provider_name = (
        provider or os.getenv("LLM_PROVIDER") or DEFAULT_LLM_PROVIDER
    ).lower()
    settings = _PROVIDER_SETTINGS.get(provider_name)
    if settings is None:
        raise ValueError(
            f"Unsupported LLM provider '{provider_name}'. "
            f"Configure LLM_PROVIDER to one of: {', '.join(sorted(_PROVIDER_SETTINGS))}."
        )

    # All supported providers expose an OpenAI-compatible chat completions API, so
    # ChatOpenAI remains a viable wrapper as long as we supply their base URL and API key.
    # Introducing providers with non-OpenAI protocols will require branching here.
    resolved_model = model or os.getenv("LLM_MODEL") or DEFAULT_LLM_MODEL
    resolved_temperature = (
        temperature
        if temperature is not None
        else _coerce_float(os.getenv("LLM_TEMPERATURE"))
    )
    if resolved_temperature is None:
        resolved_temperature = DEFAULT_LLM_TEMPERATURE

    resolved_api_key = _first_non_empty(
        api_key,
        os.getenv("LLM_API_KEY"),
        _from_env(settings.get("api_key_env")),
    )
    resolved_base_url = _first_non_empty(
        base_url,
        os.getenv("LLM_BASE_URL"),
        _from_env(settings.get("base_url_env")),
        settings.get("default_base_url"),
    )

    config: dict[str, Any] = {
        "model": resolved_model,
        "temperature": resolved_temperature,
    }

    if resolved_api_key:
        config["api_key"] = resolved_api_key
    if resolved_base_url:
        config["base_url"] = resolved_base_url

    config.update(overrides)
    return ChatOpenAI(**config)
