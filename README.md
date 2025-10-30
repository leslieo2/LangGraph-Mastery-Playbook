# LangGraph Mastery Playbook

[中文版本](README.zh.md)

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg) ![uv](https://img.shields.io/badge/uv-ready-5A45FF.svg) ![CI Friendly](https://img.shields.io/badge/ci-friendly-success.svg) ![License](https://img.shields.io/badge/license-MIT-black.svg)

**Ship LangGraph agents from day one with a runnable, stage-based curriculum.**

> 🚀 Kick off with `uv run python -m src.langgraph_learning.stage01_foundations.quickstart` to ping an LLM and run a Tavily search tool directly from your terminal.

## TL;DR

- Follow a six-stage path that upgrades your LangGraph skills from graph basics to production retrieval systems.
- Every lesson is a pure-Python module with `main()` plus artifacts for graphs, checkpoints, and streaming logs.
- Works with OpenAI, OpenRouter, DeepSeek, or any compatible endpoint through `utils.create_llm`.
- Designed for builders who prefer reproducible scripts over notebooks—ideal for demos, CI, or team onboarding.

## Quickstart Demo

Run the Stage 06 deep research lesson to see LangGraph parallelism and structured outputs in action:

```bash
uv run python -m src.langgraph_learning.stage06_production_systems.agent_with_deep_research
```

Example output:
![research ](demo.gif)

The lesson also saves a graph diagram you can reference or share:

<img src="src/langgraph_learning/stage06_production_systems/artifacts/agent_with_deep_research.png" alt="Research Agent Graph" width="50%">

## Why This Project?

- **Script-first tutorials.** Most LangGraph examples online live in Jupyter notebooks; great for reading, not ideal for reuse. Every tutorial here is a standalone Python module with a `main()` entry point and stage-specific helpers.
- **Structured learning path.** Lessons are grouped into numbered stages so you always know what to study next—from Stage 01 foundational graph skills to Stage 06 production retrieval pipelines.
- **Consistent tooling.** Shared utilities handle graph visualization, environment checks, structured output inspection, and common math tools. Less boilerplate, more focus on concepts.
- **Automation friendly.** Because everything is pure Python, you can run the entire course headlessly, integrate it into CI pipelines, or extend it with your own tests.

## Learning Roadmap

| Stage | Lessons Snapshot | Skill Gains | Est. Time |
| --- | --- | --- | --- |
| `stage01_foundations` → `quickstart`, `agent_with_tool_call`, `agent_with_router`, `agent_with_tool_router`, `agent_with_reactive_router`, `agent_with_structured_output` | Verify credentials, bind tools, branch flows, rehearse reactive tool loops, and master structured output. | Stand up `MessagesState`, detect and execute tool calls, configure conditional edges, loop on tool replays, and generate structured data with validation. | ~2.5 hrs |
| `stage02_memory_basics` → `agent_with_short_term_memory`, `agent_with_chat_summary`, `agent_with_external_short_term_memory` | Layer checkpoints, summarization, and SQLite persistence onto conversational agents. | Configure `MemorySaver`, orchestrate summary reducers, and swap durable storage with minimal code changes. | ~2 hrs |
| `stage03_state_management` → `agent_with_parallel_nodes`, `agent_with_state_reducer`, `agent_with_multiple_state`, `agent_with_pydantic_schema_constrain`, `agent_with_subgraph`, `agent_with_subgraph_memory` | Master typed state, reducers, and subgraphs for larger flows. | Parallelize with `Send`, resolve reducer conflicts, scope data per node, and isolate child graph memory. | ~3 hrs |
| `stage04_operational_control` → `agent_with_interruption`, `agent_with_validation_loop`, `agent_with_tool_approval_interrupt`, `agent_with_stream_interruption`, `agent_with_dynamic_interruption`, `agent_with_durable_execution`, `agent_with_message_filter`, `agent_with_time_travel` | Practice live debugging, guardrails, and run inspection. | Inject breakpoints, build validator loops, wrap side effects with cached tasks, control streaming updates, trim history, and fork prior runs. | ~3 hrs |
| `stage05_advanced_memory_systems` → `agent_with_structured_memory`, `agent_with_fact_collection`, `agent_with_long_term_memory`, `agent_with_multi_memory_coordination` | Build multi-layer structured memories and personalization flows. | Extract structured profiles, capture facts, author reflective summaries, and route requests across memories. | ~3 hrs |
| `stage06_production_systems` → `agent_with_parallel_retrieval`, `agent_with_semantic_memory`, `agent_with_production_memory`, `agent_with_deep_research` | Ship production-ready research and retrieval pipelines. | Orchestrate parallel retrievers, blend semantic recall, configure external checkpoint backends, and run deep research workflows. | ~3 hrs |

Every Python file begins with a “What You'll Learn / Lesson Flow” docstring so you can skim the topic before running it.

## Getting Started

<details>
<summary><b>📋 Setup Instructions (Click to expand)</b></summary>

We use [uv](https://docs.astral.sh/uv/) for dependency management; it can still export a traditional
`requirements.txt` when needed.

```bash
git clone https://github.com/leslieo2/LangGraph-Mastery-Playbook
cd LangGraph-Mastery-Playbook
uv venv              # create a virtual environment
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install .
```

> Need a pinned export? Run `uv pip compile requirements.in -o requirements.txt` to generate one.

### Environment Variables

Set the API keys required for the lessons you plan to run:

```bash
export OPENAI_API_KEY="sk-..."        # Required for all LLM demos
export TAVILY_API_KEY="tvly-..."      # Needed for Stage 06 production systems
export LANGSMITH_API_KEY="ls-..."     # Optional, enables tracing in supported lessons
```

> Stage 06's production memory demo also expects `BACKEND_KIND` (`postgres`, `mongodb`, or `redis`), a matching `BACKEND_URI`, and optional `BACKEND_INITIALIZE=true` if you want the script to create schemas on first run.

### LLM Provider Configuration

To switch models/providers, just edit the `.env` file.

Example switch to OpenRouter:

```dotenv
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-your-openrouter-key
OPENROUTER_MODEL=anthropic/claude-3-haiku
OPENROUTER_TEMPERATURE=0.2
```

</details>

<details>
<summary><b>🚀 Running Lessons (Click to expand)</b></summary>

Each script is executable via `python -m` (uv users can also run `uv run ...`):

```bash
# Stage 01 examples
python -m src.langgraph_learning.stage01_foundations.quickstart
python -m src.langgraph_learning.stage01_foundations.agent_with_tool_call
python -m src.langgraph_learning.stage01_foundations.agent_with_router
python -m src.langgraph_learning.stage01_foundations.agent_with_tool_router
python -m src.langgraph_learning.stage01_foundations.agent_with_reactive_router
python -m src.langgraph_learning.stage01_foundations.agent_with_structured_output

# Stage 02 memory basics
python -m src.langgraph_learning.stage02_memory_basics.agent_with_short_term_memory
python -m src.langgraph_learning.stage02_memory_basics.agent_with_chat_summary
python -m src.langgraph_learning.stage02_memory_basics.agent_with_external_short_term_memory

# Stage 03 state management
python -m src.langgraph_learning.stage03_state_management.agent_with_parallel_nodes
python -m src.langgraph_learning.stage03_state_management.agent_with_state_reducer
python -m src.langgraph_learning.stage03_state_management.agent_with_multiple_state
python -m src.langgraph_learning.stage03_state_management.agent_with_pydantic_schema_constrain
python -m src.langgraph_learning.stage03_state_management.agent_with_subgraph
python -m src.langgraph_learning.stage03_state_management.agent_with_subgraph_memory

# Stage 04 operational control
python -m src.langgraph_learning.stage04_operational_control.agent_with_interruption
python -m src.langgraph_learning.stage04_operational_control.agent_with_validation_loop
python -m src.langgraph_learning.stage04_operational_control.agent_with_tool_approval_interrupt
python -m src.langgraph_learning.stage04_operational_control.agent_with_stream_interruption
python -m src.langgraph_learning.stage04_operational_control.agent_with_dynamic_interruption
python -m src.langgraph_learning.stage04_operational_control.agent_with_durable_execution
python -m src.langgraph_learning.stage04_operational_control.agent_with_message_filter
python -m src.langgraph_learning.stage04_operational_control.agent_with_time_travel

# Stage 05 advanced memory systems
python -m src.langgraph_learning.stage05_advanced_memory_systems.agent_with_structured_memory
python -m src.langgraph_learning.stage05_advanced_memory_systems.agent_with_fact_collection
python -m src.langgraph_learning.stage05_advanced_memory_systems.agent_with_long_term_memory
python -m src.langgraph_learning.stage05_advanced_memory_systems.agent_with_multi_memory_coordination

# Stage 06 production systems
python -m src.langgraph_learning.stage06_production_systems.agent_with_parallel_retrieval
python -m src.langgraph_learning.stage06_production_systems.agent_with_semantic_memory
python -m src.langgraph_learning.stage06_production_systems.agent_with_production_memory
python -m src.langgraph_learning.stage06_production_systems.agent_with_deep_research
```

Most lessons generate a graph visualization (PNG) inside the module's `artifacts/` directory. Streaming lessons print incremental updates; debugging lessons may prompt for manual approval.

</details>

## Comparing to Notebook-Heavy Tutorials

- **Reproducibility:** No hidden notebook state—each run starts from a clean `main()` function.
- **Version control friendly:** Diffs stay readable, and docstring summaries keep the narrative close to the code.
- **Integration ready:** Drop modules into CI, wrap them with pytest, or use them as templates for your own LangGraph services.

If you prefer notebooks, you can still adapt these scripts into notebooks, but this project intentionally emphasizes production-style workflows.

## Contributing

- Fork the repo, create a branch per feature or lesson improvement, and submit a PR.
- Run `black .` to keep formatting consistent before committing changes.
- Ensure `python -m compileall src` passes before opening a PR.
- Add a docstring summary to any new lesson so it aligns with the staged curriculum.

Happy agent building! 🎯

## LangGraph Studio Integration

All agents in this playbook are available in **LangGraph Studio** for visual debugging, testing, and deployment. See [STUDIO_SETUP.md](STUDIO_SETUP.md) for detailed setup instructions.

**Quick Start:**
```bash
langgraph dev
```
Then open: `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`

<details>
<summary><b>🔧 Optional Dependencies & Smoke Tests (Click to expand)</b></summary>

Some production-grade lessons require extra packages:

- `langgraph-checkpoint-postgres`
- `langgraph-checkpoint-mongodb`
- `langgraph-checkpoint-redis`
- `langmem`

Install them as needed, for example:

```bash
uv pip install langgraph-checkpoint-postgres langgraph-checkpoint-mongodb \
  langgraph-checkpoint-redis langmem
```

Once your databases are available, run the smoke harness to verify connections:

```bash
POSTGRES_URI=postgresql://... \
MONGODB_URI=mongodb://... \
REDIS_URI=redis://... \
uv run python scripts/production_memory_smoke.py
```

The script prints which backends connected successfully and surfaces any issues before exercising the lesson modules.

</details>

## Acknowledgements

- The structure and many lesson ideas draw inspiration from the excellent [Intro to LangGraph](https://academy.langchain.com/courses/take/intro-to-langgraph) course.
- The official [LangChain + LangGraph documentation](https://docs.langchain.com/) remains an invaluable reference while following these scripts.
