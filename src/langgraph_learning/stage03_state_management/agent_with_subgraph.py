"""
Sub-Graph Orchestration: Specialized Pipelines with Shared State

=== PROBLEM STATEMENT ===
Cross-functional teams often need to run distinct analyses on the same data. Without clear
boundaries, one team’s schema changes can break another’s workflow or leak private fields.

=== CORE SOLUTION ===
This lesson assembles two specialized sub-graphs—failure analysis and question trends—
that plug into a parent pipeline. Each sub-graph owns its schema and publishes only the
fields the parent needs.

=== KEY INNOVATION ===
- **Schema Isolation**: Sub-graphs define their own state types to avoid accidental coupling.
- **Selective Promotion**: Parent state overlaps on shared keys but keeps private fields local.
- **Nested Compilation**: Demonstrates how compiled sub-graphs slot into a larger workflow.

=== COMPARISON WITH MONOLITHIC GRAPHS ===
| Single Monolithic Graph | Sub-Graph Architecture (this file) |
|-------------------------|------------------------------------|
| All nodes share one schema | Each team maintains its own state contract |
| Harder to evolve subsystems | Sub-graphs can change internally without breaking parent |
| No clear ownership lines    | Outputs explicitly promoted back to parent pipeline |

What You'll Learn
1. Split a LangGraph workflow into sub-graphs that manage their own state schemas, so each team can evolve independently without breaking shared contracts.
2. Share parent state via overlapping keys while keeping sub-graph outputs isolated, demonstrating how selective inheritance prevents accidental cross-talk.
3. Visualize the nested structure with xray graphs and run a deterministic demo that proves the summaries align with the designed data flow.

Lesson Flow
1. Define typed schemas for raw logs plus the two specialized sub-graphs, clarifying which fields are private versus promoted back to the parent.
2. Compile failure analysis and question summarization sub-graphs that publish targeted output, wiring their intermediate nodes to mimic production review steps.
3. Nest the compiled graphs inside a parent pipeline, render its diagram, and invoke it on sample logs to compare failure and question insights side by side.
"""

from __future__ import annotations

import operator
from typing import Annotated, Iterable

from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph

from src.langgraph_learning.utils import save_graph_image


class Log(TypedDict):
    """Raw conversation log as captured from the platform."""

    id: str
    question: str
    docs: list[str] | None
    answer: str
    grade: int | None
    grader: str | None
    feedback: str | None


class FailureAnalysisState(TypedDict):
    """State schema for nodes that prepare and summarize failures."""

    cleaned_logs: list[Log]
    failures: list[Log]
    fa_summary: str
    processed_logs: list[str]


class FailureAnalysisOutputState(TypedDict):
    """Subset of failure analysis fields exposed to the parent graph."""

    fa_summary: str
    processed_logs: list[str]


class QuestionSummarizationState(TypedDict):
    """State schema for nodes that aggregate user question topics."""

    cleaned_logs: list[Log]
    qs_summary: str
    report: str
    processed_logs: list[str]


class QuestionSummarizationOutputState(TypedDict):
    """Question summarization fields returned to the parent graph."""

    qs_summary: str
    report: str
    processed_logs: list[str]


class EntryGraphState(TypedDict):
    """Shared state flowing through the parent graph and sub-graphs."""

    raw_logs: list[Log]
    cleaned_logs: list[Log]
    fa_summary: str
    report: str
    processed_logs: Annotated[list[str], operator.add]


def summarize_failures(failures: Iterable[Log]) -> str:
    topics = {log.get("feedback") or "Unspecified issue" for log in failures}
    if not topics:
        return "No graded failures detected across recent conversations."
    return "Areas to improve: " + "; ".join(sorted(topics))


def summarize_questions(logs: Iterable[Log]) -> str:
    prompts = {log["question"] for log in logs}
    if not prompts:
        return "No user questions captured."
    return "Users asked about: " + "; ".join(sorted(prompts))


def build_failure_analysis_subgraph():
    """Compile the failure-analysis sub-graph."""

    def get_failures(state: FailureAnalysisState):
        failures = [
            log for log in state["cleaned_logs"] if log.get("grade") is not None
        ]
        return {"failures": failures}

    def generate_summary(state: FailureAnalysisState):
        failures = state["failures"]
        summary = summarize_failures(failures)
        processed = [f"failure-analysis:{log['id']}" for log in failures] or [
            "failure-analysis:none"
        ]
        return {"fa_summary": summary, "processed_logs": processed}

    builder = StateGraph(
        state_schema=FailureAnalysisState, output_schema=FailureAnalysisOutputState
    )
    builder.add_node("get_failures", get_failures)
    builder.add_node("generate_summary", generate_summary)
    builder.add_edge(START, "get_failures")
    builder.add_edge("get_failures", "generate_summary")
    builder.add_edge("generate_summary", END)
    return builder.compile()


def build_question_summary_subgraph():
    """Compile the question-summarization sub-graph."""

    def generate_summary(state: QuestionSummarizationState):
        logs = state["cleaned_logs"]
        summary = summarize_questions(logs)
        processed = [f"question-summary:{log['id']}" for log in logs] or [
            "question-summary:none"
        ]
        return {"qs_summary": summary, "processed_logs": processed}

    def send_to_slack(state: QuestionSummarizationState):
        summary = state["qs_summary"]
        report = f"Slack digest:\n{summary}"
        return {"report": report}

    builder = StateGraph(
        state_schema=QuestionSummarizationState,
        output_schema=QuestionSummarizationOutputState,
    )
    builder.add_node("generate_summary", generate_summary)
    builder.add_node("send_to_slack", send_to_slack)
    builder.add_edge(START, "generate_summary")
    builder.add_edge("generate_summary", "send_to_slack")
    builder.add_edge("send_to_slack", END)
    return builder.compile()


def build_entry_app():
    """Compile the parent graph that routes into sub-graphs."""
    failure_analysis = build_failure_analysis_subgraph()
    question_summary = build_question_summary_subgraph()

    def clean_logs(state: EntryGraphState):
        # In a real pipeline this would normalize docs, redact PII, etc.
        return {"cleaned_logs": state["raw_logs"]}

    builder = StateGraph(EntryGraphState)
    builder.add_node("clean_logs", clean_logs)
    builder.add_node("failure_analysis", failure_analysis)
    builder.add_node("question_summarization", question_summary)
    builder.add_edge(START, "clean_logs")
    builder.add_edge("clean_logs", "failure_analysis")
    builder.add_edge("clean_logs", "question_summarization")
    builder.add_edge("failure_analysis", END)
    builder.add_edge("question_summarization", END)
    return builder.compile()


def demo_logs() -> list[Log]:
    """Return deterministic sample logs for the walkthrough."""
    return [
        Log(  # type: ignore[misc]
            id="1",
            question="How can I import ChatOllama?",
            docs=["LangChain import examples"],
            answer="Use: from langchain_community.chat_models import ChatOllama.",
            grade=None,
            grader=None,
            feedback=None,
        ),
        Log(  # type: ignore[misc]
            id="2",
            question="How do I use Chroma vector store?",
            docs=["Vector store overview"],
            answer="Instantiate retriever and call create_retrieval_chain.",
            grade=0,
            grader="Document Relevance Recall",
            feedback="Retrieved docs discuss vector stores but mention nothing specific to Chroma.",
        ),
    ]


def run_demo(app) -> None:
    """Invoke the parent graph and print the merged result."""
    result = app.invoke({"raw_logs": demo_logs()})
    print("\n=== Sub-Graph Output ===")
    for key, value in result.items():
        print(f"{key}: {value}")


def main() -> None:
    app = build_entry_app()
    save_graph_image(app, filename="artifacts/agent_with_subgraph.png", xray=True)
    run_demo(app)


if __name__ == "__main__":
    main()
