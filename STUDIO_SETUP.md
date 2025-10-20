# LangGraph Studio Setup for LangGraph Mastery Playbook

This guide explains how to use LangGraph Studio to visualize, interact with, and debug all the agents in the LangGraph Mastery Playbook.

## Prerequisites

Before you begin, ensure you have:

- **LangSmith Account**: [Sign up for free](https://smith.langchain.com/settings)
- **LangSmith API Key**: Get your API key from [LangSmith settings](https://smith.langchain.com/settings)
- **Python 3.10+**: Required for LangGraph
- **Required API Keys**: OpenAI, DeepSeek, or other LLM provider keys

## Quick Start

### 1. Configure Environment

Update your `.env` file with your actual API keys:

```bash
# Replace with your actual keys
LANGSMITH_API_KEY=lsv2_your_actual_langsmith_key_here
OPENAI_API_KEY=sk-your-actual-openai-key
DEEPSEEK_API_KEY=sk-your-actual-deepseek-key
TAVILY_API_KEY=tvly-your-actual-tavily-key
```

### 2. Start LangGraph Studio

Run the following command from the project root:

```bash
langgraph dev
```

This will start the LangGraph server on `http://127.0.0.1:2024`.

### 3. Access Studio

Open your browser and navigate to:

```
https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

**Note for Safari users**: Safari blocks `localhost` connections. Use the `--tunnel` flag:

```bash
langgraph dev --tunnel
```

## Available Agents in Studio

The following agents are available in Studio, organized by learning stage:

### Stage 01: Foundations
- `stage01_tool_call` - Basic tool calling with multiply function
- `stage01_router` - Conditional routing with mood detection
- `stage01_tool_router` - Tool-based routing
- `stage01_reactive_router` - Reactive tool loops

### Stage 02: Memory Basics
- `stage02_short_term_memory` - MemorySaver checkpoints
- `stage02_chat_summary` - Conversation summarization
- `stage02_external_memory` - SQLite persistence

### Stage 03: State Management
- `stage03_parallel_nodes` - Parallel execution
- `stage03_subgraph` - Subgraph composition
- `stage03_subgraph_memory` - Isolated memory scopes
- `stage03_multiple_state` - Schema visibility patterns
- `stage03_schema_validation` - TypedDict vs dataclass vs Pydantic
- `stage03_state_reducer` - Reducer conflict resolution

### Stage 04: Operational Control
- `stage04_interruption` - Breakpoint injection
- `stage04_dynamic_interruption` - In-node interrupt guardrails
- `stage04_validation_loop` - Guardrail validation
- `stage04_tool_approval` - Tool approval workflows
- `stage04_stream_interruption` - Streaming control
- `stage04_time_travel` - Run inspection and forking
- `stage04_message_filter` - Message history trimming strategies

### Stage 05: Advanced Memory Systems
- `stage05_structured_memory` - Structured data extraction
- `stage05_fact_collection` - Fact-based memory
- `stage05_long_term_memory` - Long-term persistence
- `stage05_multi_memory` - Multi-memory routing

### Stage 06: Production Systems
- `stage06_parallel_retrieval` - Parallel search operations
- `stage06_semantic_memory` - Semantic search integration
- `stage06_production_memory` - External database backends
- `stage06_deep_research` - End-to-end research workflows

> Tip: Each Studio graph accepts `configurable.provider`, `configurable.model`, `configurable.temperature`, and (when applicable) `configurable.api_key` / `configurable.base_url`. Use these overrides in Studio to experiment with different LLM backends without editing code.

## Using Studio Features

### Visual Graph Inspection
- View the complete graph structure for each agent
- See node connections and conditional edges
- Understand state flow and message routing

### Interactive Testing
- Send messages to any agent directly from Studio
- Observe step-by-step execution
- Inspect intermediate states and tool calls

### Debugging & Tracing
- Trace every decision and tool invocation
- View exact prompts, tool arguments, and return values
- Monitor token usage and latency metrics

### Hot Reloading
- Keep your dev server running while editing code
- Studio automatically reloads when you save changes
- Re-run conversations from any step to verify behavior changes

## Configuration Details

The Studio integration is configured in `langgraph.json`:

```json
{
  "dependencies": ["."],
  "graphs": {
    "stage01_tool_call": "./src/langgraph_learning/stage01_foundations/agent_with_tool_call.py:studio_graph",
    // ... more agents
  },
  "env": ".env"
}
```

Each graph entry points to a Python function that returns a compiled LangGraph application.

## Troubleshooting

### Common Issues

1. **"Graph not found" errors**
   - Ensure the function name in `langgraph.json` matches the actual function in the Python file
   - Check that the file path is correct

2. **LangSmith connection issues**
   - Verify your `LANGSMITH_API_KEY` is correct
   - Check that `LANGSMITH_TRACING=true` is set

3. **Module import errors**
   - Run `pip install -e .` to install the package in development mode
   - Ensure all dependencies are installed from `pyproject.toml`

4. **Safari connection blocked**
   - Use `langgraph dev --tunnel` for secure tunnel access

### Development Workflow

1. Start Studio: `langgraph dev`
2. Make code changes to agents
3. Save files - Studio automatically reloads
4. Test changes directly in the Studio interface
5. Use LangSmith tracing to debug issues

## Next Steps

- Explore each agent in Studio to understand the visual representation
- Use the tracing features to debug complex agent behaviors
- Experiment with different inputs to test edge cases
- Modify agent code and observe real-time updates in Studio

For more information, refer to the [LangGraph Studio documentation](https://docs.langchain.com/langgraph/studio).
