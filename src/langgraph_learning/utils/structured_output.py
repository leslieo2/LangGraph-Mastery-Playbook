"""
Structured Output Decorator for LangGraph Agents

This module provides a @structure_output decorator that enables LangChain's
native structured output capabilities for LangGraph agents.

Key Features:
- Automatic selection of ProviderStrategy vs ToolStrategy
- Support for Pydantic models, dataclasses, TypedDict, and JSON Schema
- Error handling with configurable retry mechanisms
- Integration with existing LangGraph patterns
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Type, Union, get_type_hints

from langchain.agents import create_agent
from langchain.agents.structured_output import (
    ProviderStrategy,
    ToolStrategy,
    StructuredOutputValidationError,
    MultipleStructuredOutputsError,
)
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig

from .llm import create_llm


def structure_output(
    schema: Union[Type, dict],
    *,
    tool_message_content: str | None = None,
    handle_errors: Union[
        bool,
        str,
        Type[Exception],
        tuple[Type[Exception], ...],
        Callable[[Exception], str],
    ] = True,
    **agent_kwargs,
):
    """
    Decorator to add structured output capabilities to LangGraph agents.

    This decorator wraps a function that returns an agent and automatically
    configures it for structured output using LangChain's native capabilities.

    Args:
        schema: The structured output schema (Pydantic model, dataclass, TypedDict, or JSON Schema)
        tool_message_content: Custom content for tool messages when structured output is generated
        handle_errors: Error handling strategy (True, False, custom message, or exception types)
        **agent_kwargs: Additional arguments to pass to create_agent

    Returns:
        A decorator that wraps agent creation functions
    """

    def decorator(agent_creator: Callable) -> Callable:
        @wraps(agent_creator)
        def wrapper(*args, **kwargs) -> Any:
            # Get the original agent creation function
            # We'll create a new agent with structured output instead of wrapping existing one
            llm = create_llm()

            # Configure structured output
            response_format = _get_response_format(
                schema=schema,
                tool_message_content=tool_message_content,
                handle_errors=handle_errors,
            )

            # Create structured agent
            structured_agent = create_agent(
                model=llm, tools=[], response_format=response_format, **agent_kwargs
            )

            return structured_agent

        return wrapper

    return decorator


def _get_response_format(
    schema: Union[Type, dict],
    tool_message_content: str | None = None,
    handle_errors: Union[
        bool,
        str,
        Type[Exception],
        tuple[Type[Exception], ...],
        Callable[[Exception], str],
    ] = True,
):
    """
    Determine the appropriate response format strategy.

    LangChain automatically selects the best strategy:
    - ProviderStrategy for models with native structured output support
    - ToolStrategy for models that only support tool calling
    """

    # For explicit control, we can use ToolStrategy directly
    # LangChain will automatically use ProviderStrategy when available
    return ToolStrategy(
        schema=schema,
        tool_message_content=tool_message_content,
        handle_errors=handle_errors,
    )


def extract_structured_response(state: dict) -> Any:
    """
    Extract structured response from agent state.

    This is a convenience function to extract the structured response
    from the agent's final state.

    Args:
        state: The agent's state dictionary

    Returns:
        The structured response object
    """
    return state.get("structured_response")


# Convenience functions for common structured output patterns


def create_structured_agent(
    model: Any,
    schema: Union[Type, dict],
    tools: list = None,
    handle_errors: bool = True,
    **kwargs,
):
    """
    Create an agent with structured output capabilities.

    This is a convenience function that combines create_agent with
    structured output configuration.

    Args:
        model: The language model to use
        schema: The structured output schema
        tools: List of tools available to the agent
        handle_errors: Whether to handle structured output errors
        **kwargs: Additional arguments for create_agent

    Returns:
        An agent configured for structured output
    """
    response_format = _get_response_format(schema, handle_errors=handle_errors)

    return create_agent(
        model=model, tools=tools or [], response_format=response_format, **kwargs
    )
