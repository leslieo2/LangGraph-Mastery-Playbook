"""This LangGraph warm-up script verifies provider credentials, chats with an OpenAI model, and invokes a Tavily search tool to showcase the baseline stack.

What You'll Learn
1. Confirm the required API keys early to avoid runtime surprises.
2. Send a first message to an OpenAI chat model using LangChain abstractions.
3. Invoke a Tavily web search tool and inspect the raw document payload.

Lesson Flow
1. Validate environment variables with `require_env`.
2. Instantiate `ChatOpenAI`, create a `HumanMessage`, and observe the model reply.
3. Call `TavilySearch` with a sample query and print the returned metadata.
"""

from __future__ import annotations

from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage

from src.langgraph_learning.utils import (
    create_llm,
    require_env,
    require_llm_provider_api_key,
)


def run_basic_chat(model: str | None = None) -> None:
    """Send a single message to an OpenAI chat model."""

    llm = create_llm(model=model)
    messages = [HumanMessage(content="Who are you?", name="Leslie")]
    response = llm.invoke(messages)
    print("Model reply:", response.content)


def run_tavily_search(query: str = "What is LangGraph?") -> None:
    """Execute a Tavily web search and display the raw documents."""

    tavily_search = TavilySearch(max_results=3)
    search_docs = tavily_search.invoke(query)
    print("Tavily search results:", search_docs)


def main() -> None:
    run_basic_chat()
    run_tavily_search()


if __name__ == "__main__":
    require_llm_provider_api_key()
    require_env("TAVILY_API_KEY")
    main()
