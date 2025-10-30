# Course Lesson Review Framework

Reusable checklist for refreshing course files so documentation and code stay aligned with teaching goals.

## 1. Capture Context
- Read the existing docstring or module header to understand the lesson’s intent.
- Skim the main graph-building functions to see the actual teaching flow before making changes.

## 2. Refresh the Lesson Docstring
- Rephrase the intro using this structure (all-caps headings, equals signs, blank line after each block):
  - `=== PROBLEM STATEMENT ===` – articulate the real-world challenge the lesson addresses.
  - `=== CORE SOLUTION ===` – summarize how this file’s agent/graph solves the problem.
  - `=== KEY INNOVATION ===` – highlight 2–4 standout techniques or design choices.
  - `=== COMPARISON … ===` – add a two-column table that contrasts this lesson with a nearby stage or alternative.
  - `What You'll Learn` – list three learner outcomes in numbered format.
  - `Lesson Flow` – outline the key build steps in sequence.
- Ensure the narrative matches the actual implementation and avoids marketing fluff.

## 3. Verify Code Supports the Teaching Story
- Confirm each major function maps directly to a lesson step (e.g., persona creation, memory updates, synthesis).
- Ensure the module exports a `studio_graph` (or equivalent) entry point that routes through shared helpers. For LLM-powered lessons, prefer `llm_from_config(config)` so Studio overrides like `provider`, `model`, `temperature`, `api_key`, and `base_url` work consistently.
- Check for any dead code, unused imports, or unrelated utilities that might distract learners.
- Add concise comments only when a non-obvious teaching point needs emphasis.

## 4. Optional Enhancements
- Suggest targeted annotations (e.g., why structured output is reused, when to remove demos) that make the teaching intent clearer.
- Document these suggestions in code when they reduce future reviewer effort.

## 5. Sanity Check
- If logic changed, run a lightweight validation such as `python -m compileall <path>` or the lesson’s demo.
- Confirm git status shows only intentional changes before committing.

Use this framework as a rinse-and-repeat guide whenever reviewing or updating other course-stage files.
