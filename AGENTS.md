# Repository Guidelines

## Project Structure & Module Organization
The Python package lives in `src/langgraph_learning`, organized into numbered stage folders (`stage01_intro` … `stage08_production_ready`) plus shared helpers in `utils`. Each lesson exposes a `main()` entry point and writes optional artifacts into its own `artifacts/` subdirectory. Distribution metadata in `build/` and `LangGraph_Mastery_Playbook.egg-info/` is generated—avoid editing it manually.

## Build, Test, and Development Commands
Create an environment and install the package with uv:
```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```
Sanity-check syntax with `uv run python -m compileall src`. Run a lesson via `uv run python -m src.langgraph_learning.stage03_memory_systems.agent_with_memory`. When preparing a release wheel, use `uv build` (outputs to `build/`).

## Coding Style & Naming Conventions
Target Python 3.10+. Follow [Black](https://black.readthedocs.io/) defaults (`black .`) and keep imports sorted per Black’s heuristic. Stage modules should retain the `stage##_topic` naming and begin with a “What You’ll Learn / Lesson Flow” docstring. Prefer type hints for public functions, wrap long prompt literals at sensible boundaries, and centralize provider config through `utils.create_llm`.

## Testing Guidelines
There is no dedicated pytest suite yet; treat each lesson as its own integration test. Before pushing, run the affected modules end-to-end with representative environment variables. If you add shared utilities, add a quick smoke script under an appropriate stage and ensure `uv run python -m compileall src` stays clean.

## Commit & Pull Request Guidelines
Commit messages follow a lightweight Conventional Commit style, e.g. `feat: centralize LLM provider configuration`. Use short, descriptive scopes (`fix: stage05 trim logic`). PRs should outline the lesson or utility touched, list the commands you ran, and call out new artifacts or screenshots. Link relevant issues and remind reviewers about any API keys or configuration needed to reproduce the run.

## Security & Configuration Tips
Never commit secrets—load provider keys via environment variables or a local `.env` excluded from Git. Document any new configuration flags in `README.md` and update `utils.create_llm` defaults instead of hard-coding values inside lessons. For reproducible demos, prefer provider-agnostic prompts and handle missing keys with clear error messages.
