[English](README.md) | [中文版本](README.zh.md)

# LangGraph Mastery Playbook

An open-source, code-first curriculum for mastering LangGraph. Instead of scattered notebooks, each lesson is a runnable Python script with a clear learning objective, a documented lesson flow, and a shared utility toolkit. The result is a reproducible, test-friendly training path that scales from quickstart experiments all the way to production-ready orchestration.

## Why This Project?

- **Script-first tutorials.** Most LangGraph examples online live in Jupyter notebooks; great for reading, not ideal for reuse. Every tutorial here is a standalone Python module with a `main()` entry point and stage-specific helpers.
- **Structured learning path.** Lessons are grouped into numbered stages so you always know what to study next—from Stage 01 fundamentals to Stage 08 production patterns.
- **Consistent tooling.** Shared utilities handle graph visualization, environment checks, TrustCall inspection, and common math tools. Less boilerplate, more focus on concepts.
- **Automation friendly.** Because everything is pure Python, you can run the entire course headlessly, integrate it into CI pipelines, or extend it with your own tests.

## Learning Roadmap

| Stage | Theme | Highlights |
| --- | --- | --- |
| `stage01_intro` | Foundations | Quickstart, state graphs, tool-calling chains |
| `stage02_agent_flows` | Building Agents | Reactive loops, tool routing |
| `stage03_memory_systems` | Memory Strategies | Checkpoints, TrustCall, SQLite persistence |
| `stage04_state_management` | State Schemas | TypedDict vs dataclass vs Pydantic, reducers |
| `stage05_conversation_control` | Message Windows | Reducer filtering, selective prompts, trimming |
| `stage06_streaming_and_monitoring` | Live Runs | Streaming modes, summaries, token inspection |
| `stage07_debugging_and_iteration` | Troubleshooting | Breakpoints, dynamic interrupts, time travel |
| `stage08_production_ready` | Production Patterns | Parallel retrieval, context synthesis |

Every Python file begins with a “What You'll Learn / Lesson Flow” docstring so you can skim the topic before running it.

## Getting Started

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
export TAVILY_API_KEY="tvly-..."      # Needed for Stage 08 parallel retrieval
export LANGSMITH_API_KEY="ls-..."     # Optional, enables tracing in supported lessons
```

### LLM Provider Configuration

All lessons create chat models through `src.langgraph_learning.utils.create_llm`, which supports multiple OpenAI-compatible providers. By default, it targets OpenAI's `gpt-5-nano`, but you can switch providers by setting these environment variables:

- `LLM_PROVIDER`: one of `openai` (default), `openrouter`, or `deepseek`.
- `LLM_MODEL`: override the default model name, e.g. `gpt-4o-mini` or `openrouter/anthropic/claude-3-haiku`.
- `LLM_TEMPERATURE`: optional float override for sampling temperature.
- `LLM_API_KEY`: shared fallback for any provider if a provider-specific key isn't set.
- Provider-specific keys/base URLs:
  - OpenRouter: `OPENROUTER_API_KEY`, optional `OPENROUTER_BASE_URL` (defaults to `https://openrouter.ai/api/v1`).
  - DeepSeek: `DEEPSEEK_API_KEY`, optional `DEEPSEEK_BASE_URL` (defaults to `https://api.deepseek.com/v1`).
  - OpenAI: `OPENAI_API_KEY`, optional `OPENAI_BASE_URL`.

Example (switch to OpenRouter):

```bash
export LLM_PROVIDER="openrouter"
export OPENROUTER_API_KEY="or-..."
export LLM_MODEL="openrouter/anthropic/claude-3-haiku"
```

### Running Lessons

Each script is executable via `python -m` (uv users can also run `uv run ...`):

```bash
# Stage 01 examples
python -m src.langgraph_learning.stage01_intro.quickstart
python -m src.langgraph_learning.stage01_intro.tool_calling_chain

# Stage 03 memory systems
python -m src.langgraph_learning.stage03_memory_systems.agent_with_memory
python -m src.langgraph_learning.stage03_memory_systems.trustcall_memory_agent

# Stage 07 debugging flows
python -m src.langgraph_learning.stage07_debugging_and_iteration.breakpoints
```

Most lessons generate a graph visualization (PNG) inside the module’s `artifacts/` directory. Streaming lessons print incremental updates; debugging lessons may prompt for manual approval.

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

## Acknowledgements

- The structure and many lesson ideas draw inspiration from the excellent [Intro to LangGraph](https://academy.langchain.com/courses/take/intro-to-langgraph) course.
- The official [LangChain + LangGraph documentation](https://docs.langchain.com/) remains an invaluable reference while following these scripts.
