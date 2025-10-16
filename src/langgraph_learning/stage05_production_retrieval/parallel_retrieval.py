"""This LangGraph retrieval agent queries Tavily and Wikipedia in parallel, fuses the evidence, and prompts an LLM to synthesize an answer.

What You'll Learn
1. Orchestrate parallel retrieval pipelines (Tavily + Wikipedia) inside a single LangGraph app.
2. Merge heterogeneous context sources and feed the combined evidence into an LLM for synthesis.
3. Validate environment prerequisites and capture graph diagrams for production documentation.

Lesson Flow
1. Define the retrieval state schema with an additive context reducer.
2. Add parallel search nodes, a generation node, and connect edges for convergence at the end node.
3. Compile the graph, ensure required API keys exist, render the diagram, and execute a sample question.
"""

from __future__ import annotations

import json as json_module
import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_community.document_loaders import WikipediaLoader
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph

from src.langgraph_learning.utils import (
    create_llm,
    maybe_enable_langsmith,
    require_env,
    save_graph_image,
)


class RetrievalState(TypedDict):
    question: str
    answer: str
    context: Annotated[list[str], operator.add]


def _extract_tavily_results(raw: Any) -> list[dict[str, Any]]:
    """Normalize Tavily tool output into a list of result dictionaries."""
    if isinstance(raw, ToolMessage):
        raw = raw.content
    if isinstance(raw, str):
        try:
            raw = json_module.loads(raw)
        except ValueError:
            return []
    if isinstance(raw, dict):
        raw = raw.get("results", [])
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    return []


def build_parallel_retrieval_graph() -> StateGraph:
    """Wire two retrievers in parallel and merge their outputs for the LLM."""
    llm = create_llm()
    web_search = TavilySearch(max_results=3)

    def search_web(state: RetrievalState) -> dict[str, list[str]]:
        """Fetch fresh web snippets for the question."""
        raw_results = web_search.invoke(state["question"])
        results = _extract_tavily_results(raw_results)
        if not results:
            return {"context": []}
        formatted = "\n\n---\n\n".join(
            f'<Document href="{doc.get("url", "")}" title="{doc.get("title", "")}">\n'
            f'{doc.get("content") or doc.get("raw_content", "")}\n</Document>'
            for doc in results
        )
        return {"context": [formatted] if formatted else []}

    def search_wikipedia(state: RetrievalState) -> dict[str, list[str]]:
        """Pull a couple of Wikipedia pages about the topic."""
        docs = WikipediaLoader(query=state["question"], load_max_docs=2).load()
        formatted = "\n\n---\n\n".join(
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}">\n'
            f"{doc.page_content}\n</Document>"
            for doc in docs
        )
        return {"context": [formatted]}

    def generate_answer(state: RetrievalState):
        """Ask the LLM to synthesize a reply from combined context."""
        context_block = "\n\n".join(state["context"])
        instructions = (
            "Answer the user's question using the provided context.\n"
            f"Question: {state['question']}\n"
            f"Context:\n{context_block}"
        )
        response = llm.invoke(
            [
                SystemMessage(content=instructions),
                HumanMessage(content="Provide the answer."),
            ]
        )
        return {"answer": response}

    builder = StateGraph(RetrievalState)
    builder.add_node("search_web", search_web)
    builder.add_node("search_wikipedia", search_wikipedia)
    builder.add_node("generate_answer", generate_answer)

    builder.add_edge(START, "search_web")
    builder.add_edge(START, "search_wikipedia")
    builder.add_edge("search_web", "generate_answer")
    builder.add_edge("search_wikipedia", "generate_answer")
    builder.add_edge("generate_answer", END)

    graph = builder.compile()
    save_graph_image(graph, filename="artifacts/parallel_retrieval.png")
    return graph


def run_demo(graph: StateGraph) -> None:
    question = "How were Nvidia's Q2 2024 earnings?"
    print(f"\n--- Running graph for: {question} ---")
    result = graph.invoke({"question": question})
    answer = result["answer"]
    content = getattr(answer, "content", answer)
    print("\n--- Model answer ---")
    print(content)


def main() -> None:
    require_env("OPENAI_API_KEY")
    require_env("TAVILY_API_KEY")
    maybe_enable_langsmith()
    graph = build_parallel_retrieval_graph()
    run_demo(graph)


if __name__ == "__main__":
    main()
