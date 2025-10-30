"""
Structured Output with LangChain's Native Support

=== WHY THIS COURSE MATTERS ===
Getting LLMs to output structured data reliably is essential for production systems.
Traditional approaches have limitations:

1. **Schema Compliance**: LLMs often produce malformed JSON or miss required fields
2. **Data Loss**: Regenerating entire schemas wastes tokens and loses context
3. **Complex Schema Failure**: Nested structures frequently cause parsing errors

LangChain's native structured output solves these problems by:
- Using provider-native structured output when available (OpenAI, Grok)
- Falling back to tool calling for other models
- Providing automatic validation and error handling

=== Core Concepts ===
1. **ProviderStrategy**: Uses model-native structured output (most reliable)
2. **ToolStrategy**: Uses tool calling for structured output (universal support)
3. **Automatic Selection**: LangChain chooses the best strategy automatically
4. **Error Handling**: Built-in retry mechanisms for validation failures

=== Key Innovation ===
LangChain's structured output:
- Provides native integration with model providers
- Offers automatic fallback strategies
- Supports multiple schema types (Pydantic, dataclasses, TypedDict, JSON Schema)
- Includes comprehensive error handling

=== Learning Objectives ===
1. Understand LangChain's structured output strategies
2. Learn to use the @structure_output decorator
3. Compare ProviderStrategy vs ToolStrategy
4. Implement error handling for structured output
"""

from __future__ import annotations

from typing import List, Literal

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.append(project_root)

from src.langgraph_learning.utils import (
    create_llm,
    llm_from_config,
    maybe_enable_langsmith,
    require_llm_provider_api_key,
)
from src.langgraph_learning.utils.structured_output import (
    structure_output,
    create_structured_agent,
    extract_structured_response,
)


# =============================================================================
# 1. DEFINE STRUCTURED OUTPUT SCHEMAS
# These define the data structures we want the agent to output
# =============================================================================


class ContactInfo(BaseModel):
    """Contact information extracted from text."""

    name: str = Field(description="The person's full name")
    email: str = Field(description="The email address")
    phone: str = Field(description="The phone number")


class ProductReview(BaseModel):
    """Analysis of a product review."""

    rating: int | None = Field(description="The rating (1-5)", ge=1, le=5)
    sentiment: Literal["positive", "negative"] = Field(description="Overall sentiment")
    key_points: List[str] = Field(description="Key points mentioned in the review")


class MeetingAction(BaseModel):
    """Action items extracted from meeting notes."""

    task: str = Field(description="The specific task to be completed")
    assignee: str = Field(description="Person responsible for the task")
    priority: Literal["low", "medium", "high"] = Field(description="Priority level")


# =============================================================================
# 2. AGENT IMPLEMENTATIONS WITH STRUCTURED OUTPUT
# Demonstrating different approaches to structured output
# =============================================================================


def create_contact_extractor_agent():
    """Create an agent that extracts contact information using structured output."""
    llm = create_llm()

    # Method 1: Using the @structure_output decorator
    @structure_output(ContactInfo)
    def create_agent_with_decorator():
        return create_agent(model=llm, tools=[])

    return create_agent_with_decorator()


def create_review_analyzer_agent():
    """Create an agent that analyzes product reviews with error handling."""
    llm = create_llm()

    # Method 2: Using explicit ToolStrategy with custom error handling
    response_format = ToolStrategy(
        schema=ProductReview,
        tool_message_content="Review analysis completed!",
        handle_errors="Please provide a valid rating between 1-5 and include key points.",
    )

    return create_agent(
        model=llm,
        tools=[],
        response_format=response_format,
        system_prompt="You are a helpful assistant that analyzes product reviews.",
    )


def create_meeting_parser_agent():
    """Create an agent that extracts meeting action items."""
    llm = create_llm()

    # Method 3: Using the convenience function
    return create_structured_agent(
        model=llm,
        schema=MeetingAction,
        tools=[],
        system_prompt="Extract action items from meeting notes.",
    )


# =============================================================================
# 3. DEMONSTRATION FUNCTIONS
# Show how structured output works in practice
# =============================================================================


def demo_contact_extraction():
    """Demonstrate contact information extraction."""
    print("=== DEMO 1: Contact Information Extraction ===\n")

    agent = create_contact_extractor_agent()

    # Test input
    test_input = "John Doe, john@example.com, (555) 123-4567"
    print(f"Input: {test_input}")

    result = agent.invoke(
        {"messages": [HumanMessage(f"Extract contact info from: {test_input}")]}
    )

    structured_response = extract_structured_response(result)
    print(f"Structured Output: {structured_response}")
    print(f"Type: {type(structured_response)}")

    # Access structured data
    if structured_response:
        print(f"Name: {structured_response.name}")
        print(f"Email: {structured_response.email}")
        print(f"Phone: {structured_response.phone}")


def demo_review_analysis():
    """Demonstrate product review analysis with error handling."""
    print("\n=== DEMO 2: Product Review Analysis ===\n")

    agent = create_review_analyzer_agent()

    # Test input with rating outside bounds
    test_input = "Amazing product: 10/10! Fast shipping, great quality, but expensive."
    print(f"Input: {test_input}")

    result = agent.invoke(
        {"messages": [HumanMessage(f"Analyze this review: {test_input}")]}
    )

    structured_response = extract_structured_response(result)
    print(f"Structured Output: {structured_response}")

    # Show how error handling worked
    if structured_response:
        print(f"Rating: {structured_response.rating}")
        print(f"Sentiment: {structured_response.sentiment}")
        print(f"Key Points: {structured_response.key_points}")


def demo_meeting_actions():
    """Demonstrate meeting action item extraction."""
    print("\n=== DEMO 3: Meeting Action Items ===\n")

    agent = create_meeting_parser_agent()

    # Test input
    test_input = "Sarah needs to update the project timeline as soon as possible"
    print(f"Input: {test_input}")

    result = agent.invoke(
        {"messages": [HumanMessage(f"Extract action items from: {test_input}")]}
    )

    structured_response = extract_structured_response(result)
    print(f"Structured Output: {structured_response}")

    if structured_response:
        print(f"Task: {structured_response.task}")
        print(f"Assignee: {structured_response.assignee}")
        print(f"Priority: {structured_response.priority}")


def demo_error_handling():
    """Demonstrate structured output error handling."""
    print("\n=== DEMO 4: Error Handling ===\n")

    llm = create_llm()

    # Agent with strict validation
    response_format = ToolStrategy(
        schema=ProductReview, handle_errors=True  # Auto-retry on validation errors
    )

    agent = create_agent(model=llm, tools=[], response_format=response_format)

    # This should trigger error handling due to missing rating
    test_input = "Great product, loved the quality!"
    print(f"Input: {test_input}")

    result = agent.invoke({"messages": [HumanMessage(f"Analyze: {test_input}")]})

    structured_response = extract_structured_response(result)
    print(f"Final Output: {structured_response}")


# =============================================================================
# 4. MAIN DEMONSTRATION
# =============================================================================


def main() -> None:
    """Run all structured output demonstrations."""
    require_llm_provider_api_key()
    maybe_enable_langsmith()

    print("=" * 60)
    print("STRUCTURED OUTPUT WITH LANGCHAIN")
    print("=" * 60)
    print(
        "\nThis lesson demonstrates LangChain's native structured output capabilities."
    )
    print("Key features:")
    print("• Automatic ProviderStrategy/ToolStrategy selection")
    print("• Support for Pydantic models, dataclasses, TypedDict")
    print("• Built-in error handling and validation")
    print("• @structure_output decorator for easy integration")
    print()

    # Run demonstrations
    demo_contact_extraction()
    demo_review_analysis()
    demo_meeting_actions()
    demo_error_handling()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("1. LangChain's structured output provides reliable schema compliance")
    print("2. Automatic strategy selection ensures broad model compatibility")
    print("3. Built-in error handling reduces manual validation code")
    print("4. Multiple schema types offer flexibility for different use cases")
    print("5. The @structure_output decorator simplifies integration")


if __name__ == "__main__":
    main()


def studio_graph(config: RunnableConfig | None = None):
    """Studio entry point for the structured output lesson."""
    _, overrides = llm_from_config(config)

    # Return a simple agent for Studio demonstration
    return create_contact_extractor_agent()
