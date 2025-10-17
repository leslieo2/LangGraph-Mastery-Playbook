"""Helpers for constructing ChatOpenAI clients with consistent configuration.

Environment variables resolve in this order:
1. Explicit arguments passed to `create_llm`.
2. Provider-specific variables loaded from `.env`
3. Built-in defaults for the selected provider.
"""

from __future__ import annotations

import os
from typing import Any, Callable

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

_PROVIDER_SETTINGS: dict[str, dict[str, Any]] = {
    "openai": {
        "is_default": True,
        "env": {
            "api_key": "OPENAI_API_KEY",
            "base_url": "OPENAI_BASE_URL",
            "model": "OPENAI_MODEL",
            "temperature": "OPENAI_TEMPERATURE",
        },
        "defaults": {
            "base_url": None,
            "model": "gpt-5-nano",
            "temperature": 0.0,
        },
    },
    "openrouter": {
        "env": {
            "api_key": "OPENROUTER_API_KEY",
            "base_url": "OPENROUTER_BASE_URL",
            "model": "OPENROUTER_MODEL",
            "temperature": "OPENROUTER_TEMPERATURE",
        },
        "defaults": {
            "base_url": "https://openrouter.ai/api/v1",
            "model": "anthropic/claude-haiku-4.5",
            "temperature": 0.0,
        },
    },
    "deepseek": {
        "env": {
            "api_key": "DEEPSEEK_API_KEY",
            "base_url": "DEEPSEEK_BASE_URL",
            "model": "DEEPSEEK_MODEL",
            "temperature": "DEEPSEEK_TEMPERATURE",
        },
        "defaults": {
            "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-chat",
            "temperature": 0.0,
        },
    },
}


def _from_env(var_name: str | None) -> str | None:
    """Get environment variable value if variable name is provided.

    Args:
        var_name: Environment variable name to retrieve

    Returns:
        Environment variable value or None if not set or var_name is None
    """
    return os.getenv(var_name) if var_name else None


def _coerce_float(value: str | None) -> float | None:
    """Convert string value to float, returning None on failure.

    Args:
        value: String value to convert to float

    Returns:
        Float value or None if conversion fails
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_value(
    explicit: Any,
    env_var: str | None,
    default: Any,
    *,
    transform: Callable[[str], Any] | None = None,
) -> Any:
    """Resolve a configuration value using priority order.

    Priority order:
    1. Explicit value passed to function
    2. Environment variable value
    3. Default value

    Args:
        explicit: Explicitly provided value (highest priority)
        env_var: Environment variable name to check
        default: Default value to use if no other sources provide a value
        transform: Optional transformation function for environment values

    Returns:
        Resolved configuration value
    """
    if explicit is not None:
        return explicit

    env_value_raw = _from_env(env_var)
    if env_value_raw is not None:
        if transform is None:
            return env_value_raw
        transformed = transform(env_value_raw)
        if transformed is not None:
            return transformed

    return default


def _default_provider() -> str:
    """Determine the default LLM provider from configuration.

    Returns:
        Name of the default provider, or first configured provider if none marked default
    """
    for name, settings in _PROVIDER_SETTINGS.items():
        if settings.get("is_default"):
            return name
    # Fall back to the first configured provider if none is flagged default.
    return next(iter(_PROVIDER_SETTINGS))


def create_llm(
    *,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    **overrides: Any,
) -> ChatOpenAI:
    """Return a ChatOpenAI instance using shared defaults and provider-specific config.

    This function creates a ChatOpenAI client configured for the specified provider.
    It resolves configuration using the following priority order:
    1. Explicit function arguments
    2. Environment variables
    3. Provider defaults

    Args:
        provider: LLM provider name (openai, openrouter, deepseek)
        model: Model name to use
        temperature: Temperature parameter for generation
        api_key: API key for the provider
        base_url: Base URL for the API endpoint
        **overrides: Additional parameters to pass to ChatOpenAI

    Returns:
        Configured ChatOpenAI instance

    Raises:
        ValueError: If provider is unsupported or required model is not configured
    """
    provider_name = (
        provider or os.getenv("LLM_PROVIDER") or _default_provider()
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
    env_names = settings["env"]
    defaults = settings["defaults"]

    # Resolve configuration values
    resolved_model = _resolve_value(
        model,
        env_names.get("model"),
        defaults.get("model"),
    )
    if resolved_model is None:
        raise ValueError(
            f"No default model configured for provider '{provider_name}'. "
            "Specify a model explicitly when calling create_llm."
        )

    resolved_temperature = (
        _resolve_value(
            temperature,
            env_names.get("temperature"),
            defaults.get("temperature", 0.0),
            transform=_coerce_float,
        )
        or 0.0
    )

    resolved_api_key = _resolve_value(
        api_key,
        env_names.get("api_key"),
        defaults.get("api_key"),
    )
    resolved_base_url = _resolve_value(
        base_url,
        env_names.get("base_url"),
        defaults.get("base_url"),
    )

    # Build configuration dictionary
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


def require_llm_provider_api_key(provider: str | None = None) -> None:
    """Ensure the active LLM provider has credentials configured.

    This function checks that the specified provider has a valid API key
    configured either through environment variables or default settings.

    Args:
        provider: LLM provider name to check (uses default if None)

    Raises:
        ValueError: If provider is unsupported
        RuntimeError: If no API key is configured for the provider
    """
    provider_name = (
        provider or os.getenv("LLM_PROVIDER") or _default_provider()
    ).lower()
    settings = _PROVIDER_SETTINGS.get(provider_name)
    if settings is None:
        raise ValueError(
            f"Unsupported LLM provider '{provider_name}'. "
            f"Configure LLM_PROVIDER to one of: {', '.join(sorted(_PROVIDER_SETTINGS))}."
        )

    env_names = settings["env"]
    defaults = settings["defaults"]

    api_key_env = env_names.get("api_key")
    default_api_key = defaults.get("api_key")

    # Check if API key is available through defaults
    if default_api_key:
        return

    # Check if API key is available through environment variable
    if api_key_env and os.getenv(api_key_env):
        return

    raise RuntimeError(
        f"{api_key_env or 'API key'} must be set for provider '{provider_name}'. "
        "Set the environment variable or pass `api_key` to `create_llm`."
    )
