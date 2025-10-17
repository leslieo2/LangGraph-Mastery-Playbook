"""This LangGraph research assistant orchestrates analyst planning, multi-turn expert
interviews, and report synthesis for an end-to-end retrieval workflow.

What You'll Learn
1. Nest LangGraph sub-graphs to conduct parallel, multi-turn interviews with shared tools.
2. Gate execution with a human-in-the-loop checkpoint that can reshape downstream work.
3. Reduce analyst memos into a polished report complete with introduction, insights, and conclusion.

Lesson Flow
1. Generate analyst personas from a topic and optionally loop for editorial feedback.
2. Launch a shared interview sub-graph that searches Tavily and Wikipedia in parallel per persona.
3. Summarize all sections into a cohesive report and persist graph diagrams for documentation.
"""

from __future__ import annotations

import json as json_module
import operator
from typing import Annotated, Any, Iterable, Sequence

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    get_buffer_string,
)
from langchain_community.document_loaders import WikipediaLoader
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from src.langgraph_learning.utils import (
    create_llm,
    maybe_enable_langsmith,
    require_env,
    require_llm_provider_api_key,
    save_graph_image,
)


class Analyst(BaseModel):
    """Persona describing the point-of-view for an analyst node."""

    affiliation: str = Field(description="Primary affiliation of the analyst.")
    name: str = Field(description="Name of the analyst.")
    role: str = Field(description="Role of the analyst in the context of the topic.")
    description: str = Field(
        description="Focus, concerns, and motives for the analyst."
    )

    @property
    def persona(self) -> str:
        return (
            f"Name: {self.name}\n"
            f"Role: {self.role}\n"
            f"Affiliation: {self.affiliation}\n"
            f"Description: {self.description}\n"
        )


class Perspectives(BaseModel):
    analysts: list[Analyst] = Field(
        description="Comprehensive list of analysts with their roles and affiliations."
    )


class SearchQuery(BaseModel):
    search_query: str = Field(
        None, description="Search query derived from the ongoing interview."
    )


class InterviewState(MessagesState):
    max_num_turns: int
    context: Annotated[list[str], operator.add]
    analyst: Analyst
    interview: str
    sections: Annotated[list[str], operator.add]


class ResearchGraphState(TypedDict, total=False):
    topic: str
    max_analysts: int
    human_analyst_feedback: str | None
    analysts: list[Analyst]
    sections: Annotated[list[str], operator.add]
    introduction: str
    content: str
    conclusion: str
    final_report: str


ANALYST_INSTRUCTIONS = """You are tasked with creating a set of analyst personas.

1. Review the research topic:
{topic}

2. Examine optional editorial feedback that should inform persona selection:
{human_analyst_feedback}

3. Pick the most compelling {max_analysts} themes.

4. Assign one analyst to each theme.
"""

QUESTION_INSTRUCTIONS = """You are an analyst preparing to interview an expert.

Your goal is to surface interesting, specific insights related to your focus area.

Here are your persona details:
{goals}

Begin by introducing yourself in character, pose a targeted question, and continue
asking follow ups until you conclude with: "Thank you so much for your help!"
"""

SEARCH_INSTRUCTIONS = SystemMessage(
    content=(
        "You will be given a conversation between an analyst and an expert.\n\n"
        "Your goal is to generate a well-structured search query to support the next answer.\n"
        "Analyse the full conversation, pay attention to the latest analyst question, "
        "and convert it into a focused search query."
    )
)

ANSWER_INSTRUCTIONS = """You are an expert being interviewed by an analyst.

Persona details:
{goals}

Use only this context when answering:
{context}

Guidelines:
1. Ground every statement in the provided context.
2. Add inline citations like [1] that correspond to the numbered sources.
3. List sources at the end of your answer, preserving their order and without duplication.
"""

SECTION_WRITER_INSTRUCTIONS = """You are an expert technical writer.

Craft a short markdown section based on the analyst's interview focus:
{focus}

Structure:
## Compelling Title
### Summary
### Sources

Summaries should highlight novel findings (≤ 400 words) and retain citations as [1], [2], etc.
Combine duplicate sources so each appears once. Use the provided context to ground the memo.
"""

REPORT_WRITER_INSTRUCTIONS = """You are a technical writer creating a report on:
{topic}

You will receive analyst memos. Consolidate them into a cohesive markdown report:
1. Start with a single title header: ## Insights
2. No preamble or analyst names.
3. Preserve citations and build a deduplicated ## Sources section at the end.

Memos:
{context}
"""

INTRO_CONCLUSION_INSTRUCTIONS = """You are finishing a report on {topic}.

You will receive all memo sections.
Write either an introduction or conclusion (prompt will specify which).

Guidelines:
- No preamble beyond the requested header.
- Target ~100 words.
- Introduction: add a title (# Header) followed by ## Introduction.
- Conclusion: use ## Conclusion.

Sections:
{formatted_str_sections}
"""


def build_interview_app(
    *,
    max_interview_turns: int,
    model: str | None = None,
) -> StateGraph:
    """Compile the interview sub-graph that powers each analyst conversation."""
    llm = create_llm(model=model, temperature=0)
    tavily_search = TavilySearch(max_results=3)

    def _extract_tavily_results(raw: Any) -> list[dict[str, Any]]:
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

    def generate_question(state: InterviewState) -> dict[str, Iterable[AIMessage]]:
        persona_instructions = QUESTION_INSTRUCTIONS.format(
            goals=state["analyst"].persona
        )
        question = llm.invoke(
            [SystemMessage(content=persona_instructions), *state["messages"]]
        )
        return {"messages": [question]}

    def _format_context(chunks: Sequence[str]) -> str:
        return "\n\n---\n\n".join(chunks)

    def search_with_tavily(state: InterviewState) -> dict[str, list[str]]:
        structured_llm = llm.with_structured_output(SearchQuery)
        search_query = structured_llm.invoke([SEARCH_INSTRUCTIONS, *state["messages"]])
        raw_results = tavily_search.invoke(search_query.search_query)
        results = _extract_tavily_results(raw_results)
        if not results:
            return {"context": []}
        formatted = [
            f'<Document href="{doc.get("url", "")}" title="{doc.get("title", "")}">\n'
            f'{doc.get("content") or doc.get("raw_content", "")}\n</Document>'
            for doc in results
        ]
        combined = _format_context([doc for doc in formatted if doc.strip()])
        return {"context": [combined] if combined else []}

    def search_wikipedia(state: InterviewState) -> dict[str, list[str]]:
        structured_llm = llm.with_structured_output(SearchQuery)
        search_query = structured_llm.invoke([SEARCH_INSTRUCTIONS, *state["messages"]])
        docs = WikipediaLoader(query=search_query.search_query, load_max_docs=2).load()
        formatted = [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}">\n'
            f"{doc.page_content}\n</Document>"
            for doc in docs
        ]
        return {"context": [_format_context(formatted)]}

    def generate_answer(state: InterviewState) -> dict[str, Iterable[AIMessage]]:
        persona = state["analyst"].persona
        context = "\n\n".join(state["context"])
        system_message = ANSWER_INSTRUCTIONS.format(goals=persona, context=context)
        answer = llm.invoke([SystemMessage(content=system_message), *state["messages"]])
        answer.name = "expert"
        return {"messages": [answer]}

    def save_interview(state: InterviewState) -> dict[str, str]:
        transcript = get_buffer_string(state["messages"])
        return {"interview": transcript}

    def write_section(state: InterviewState) -> dict[str, list[str]]:
        persona_focus = state["analyst"].description
        context = "\n\n".join(state["context"])
        system_message = SECTION_WRITER_INSTRUCTIONS.format(focus=persona_focus)
        section = llm.invoke(
            [
                SystemMessage(content=system_message),
                HumanMessage(content=f"Use these materials:\n{context}"),
            ]
        )
        return {"sections": [section.content]}

    def route_messages(state: InterviewState, name: str = "expert") -> str:
        max_turns = state.get("max_num_turns", max_interview_turns)
        messages = state["messages"]
        num_responses = len(
            [m for m in messages if isinstance(m, AIMessage) and m.name == name]
        )
        if num_responses >= max_turns:
            return "save_interview"
        if len(messages) >= 2:
            last_question = messages[-2]
            if (
                isinstance(last_question, AIMessage)
                and "Thank you so much for your help" in last_question.content
            ):
                return "save_interview"
        return "ask_question"

    interview_builder = StateGraph(InterviewState)
    interview_builder.add_node("ask_question", generate_question)
    interview_builder.add_node("search_tavily", search_with_tavily)
    interview_builder.add_node("search_wikipedia", search_wikipedia)
    interview_builder.add_node("answer_question", generate_answer)
    interview_builder.add_node("save_interview", save_interview)
    interview_builder.add_node("write_section", write_section)

    interview_builder.add_edge(START, "ask_question")
    interview_builder.add_edge("ask_question", "search_tavily")
    interview_builder.add_edge("ask_question", "search_wikipedia")
    interview_builder.add_edge("search_tavily", "answer_question")
    interview_builder.add_edge("search_wikipedia", "answer_question")
    interview_builder.add_conditional_edges(
        "answer_question", route_messages, ["ask_question", "save_interview"]
    )
    interview_builder.add_edge("save_interview", "write_section")
    interview_builder.add_edge("write_section", END)

    return interview_builder.compile(checkpointer=MemorySaver()).with_config(
        run_name="conduct_interview"
    )


def build_research_assistant_graph(
    *,
    model: str | None = None,
    max_interview_turns: int = 2,
) -> StateGraph:
    """Create the full research assistant LangGraph application."""
    llm = create_llm(model=model, temperature=0)
    interview_app = build_interview_app(
        model=model, max_interview_turns=max_interview_turns
    )

    def create_analysts(state: ResearchGraphState) -> dict[str, list[Analyst]]:
        structured = llm.with_structured_output(Perspectives)
        instructions = ANALYST_INSTRUCTIONS.format(
            topic=state["topic"],
            human_analyst_feedback=state.get("human_analyst_feedback")
            or "None provided.",
            max_analysts=state["max_analysts"],
        )
        analysts = structured.invoke(
            [
                SystemMessage(content=instructions),
                HumanMessage(content="Generate the analysts."),
            ]
        )
        return {"analysts": analysts.analysts}

    def human_feedback(_: ResearchGraphState) -> None:
        return None

    def initiate_all_interviews(state: ResearchGraphState):
        feedback = state.get("human_analyst_feedback")
        if feedback:
            return "create_analysts"
        topic = state["topic"]
        return [
            Send(
                "conduct_interview",
                {
                    "analyst": analyst,
                    "messages": [
                        HumanMessage(
                            content=f"So you mentioned you were researching {topic}? Let's dive in."
                        )
                    ],
                    "max_num_turns": max_interview_turns,
                },
            )
            for analyst in state["analysts"]
        ]

    def write_report(state: ResearchGraphState) -> dict[str, str]:
        sections = state.get("sections", [])
        formatted_sections = "\n\n".join(sections)
        system_message = REPORT_WRITER_INSTRUCTIONS.format(
            topic=state["topic"], context=formatted_sections
        )
        report = llm.invoke(
            [
                SystemMessage(content=system_message),
                HumanMessage(content="Synthesize the analyst memos."),
            ]
        )
        return {"content": report.content}

    def _write_intro_or_conclusion(state: ResearchGraphState, section: str) -> str:
        sections = state.get("sections", [])
        formatted_sections = "\n\n".join(sections)
        instructions = INTRO_CONCLUSION_INSTRUCTIONS.format(
            topic=state["topic"], formatted_str_sections=formatted_sections
        )
        response = llm.invoke(
            [
                SystemMessage(content=instructions),
                HumanMessage(content=f"Write the report {section}."),
            ]
        )
        return response.content

    def write_introduction(state: ResearchGraphState) -> dict[str, str]:
        return {"introduction": _write_intro_or_conclusion(state, "introduction")}

    def write_conclusion(state: ResearchGraphState) -> dict[str, str]:
        return {"conclusion": _write_intro_or_conclusion(state, "conclusion")}

    def finalize_report(state: ResearchGraphState) -> dict[str, str]:
        body = state.get("content", "")
        intro = state.get("introduction", "")
        conclusion = state.get("conclusion", "")
        sources_block: str | None = None
        if "## Sources" in body:
            try:
                body, sources_block = body.split("\n## Sources\n", maxsplit=1)
            except ValueError:
                sources_block = None
        cleaned_body = body.lstrip("# ").strip()
        report_parts = [
            intro.strip(),
            "---",
            cleaned_body.strip(),
            "---",
            conclusion.strip(),
        ]
        report = "\n\n".join(part for part in report_parts if part)
        if sources_block:
            report += "\n\n## Sources\n" + sources_block.strip()
        return {"final_report": report}

    builder = StateGraph(ResearchGraphState)
    builder.add_node("create_analysts", create_analysts)
    builder.add_node("human_feedback", human_feedback)
    builder.add_node("conduct_interview", interview_app)
    builder.add_node("write_report", write_report)
    builder.add_node("write_introduction", write_introduction)
    builder.add_node("write_conclusion", write_conclusion)
    builder.add_node("finalize_report", finalize_report)

    builder.add_edge(START, "create_analysts")
    builder.add_edge("create_analysts", "human_feedback")
    builder.add_conditional_edges(
        "human_feedback",
        initiate_all_interviews,
        ["create_analysts", "conduct_interview"],
    )
    builder.add_edge("conduct_interview", "write_report")
    builder.add_edge("conduct_interview", "write_introduction")
    builder.add_edge("conduct_interview", "write_conclusion")
    for node in ("write_report", "write_introduction", "write_conclusion"):
        builder.add_edge(node, "finalize_report")
    builder.add_edge("finalize_report", END)

    return builder.compile(
        interrupt_before=["human_feedback"],
        checkpointer=MemorySaver(),
    )


def _print_analysts(analysts: Sequence[Analyst]) -> None:
    for idx, analyst in enumerate(analysts, start=1):
        print(f"[{idx}] {analyst.name} — {analyst.role} ({analyst.affiliation})")
        print(analyst.description)
        print("-" * 50)


def run_demo(graph: StateGraph) -> None:
    """Execute a sample topic to showcase the research workflow end-to-end."""
    topic = "The benefits of adopting LangGraph as an agent framework"
    max_analysts = 3
    feedback = "Add an analyst who runs a generative AI native startup."

    thread = {"configurable": {"thread_id": "stage05-research-demo"}}
    print(f"\n=== Launching research assistant for topic: {topic} ===")

    for event in graph.stream(
        {"topic": topic, "max_analysts": max_analysts},
        thread,
        stream_mode="values",
    ):
        analysts = event.get("analysts")
        if analysts:
            print("\nDraft analyst roster:")
            _print_analysts(analysts)

    print("\nApplying editorial feedback...")
    graph.update_state(
        thread,
        {"human_analyst_feedback": feedback},
        as_node="human_feedback",
    )

    for event in graph.stream(None, thread, stream_mode="values"):
        analysts = event.get("analysts")
        if analysts:
            print("\nUpdated analyst roster:")
            _print_analysts(analysts)

    print("\nProceeding with interviews...")
    graph.update_state(
        thread,
        {"human_analyst_feedback": None},
        as_node="human_feedback",
    )

    for update in graph.stream(None, thread, stream_mode="updates"):
        node_name = next(iter(update))
        print(f"→ {node_name}")

    final_state = graph.get_state(thread)
    report = final_state.values.get("final_report")
    if report:
        print("\n=== Final Research Report ===")
        print(report)


def main() -> None:
    require_llm_provider_api_key()
    require_env("TAVILY_API_KEY")
    maybe_enable_langsmith(project="langgraph-research-assistant")
    graph = build_research_assistant_graph()
    save_graph_image(
        graph, filename="artifacts/agent_with_deep_research.png", xray=True
    )
    run_demo(graph)


if __name__ == "__main__":
    main()
