[English](README.md) | 中文版本

# LangGraph 精通实战手册

一个开源、以代码为先的 LangGraph 自学课程。我们摒弃松散的 Notebook 片段，将每堂课都整理成可直接运行的 Python 脚本，配有明确的学习目标、清晰的课程流程说明，以及统一维护的工具库。由此得到的学习路径可复现、易于测试，既适合快速上手，也能扩展到生产级的编排实践。

## 项目亮点

- **脚本优先的教程。** 大多数 LangGraph 示例都放在 Jupyter Notebook 里——读起来方便，复用起来麻烦。本项目的每个示例都是带 `main()` 入口的独立 Python 模块，并按阶段提供专属辅助工具。
- **有结构的学习路线。** 课程分为编号的阶段，从 Stage 01 基础到 Stage 08 生产模式，每一步都知道接下来该学什么。
- **一致的工具体系。** 共享工具负责图形可视化、环境检查、TrustCall 分析，以及常用的数学工具。减少样板代码，把时间留给核心概念。
- **便于自动化。** 全部基于纯 Python，你可以无界面运行整套课程，集成到 CI 流水线，或扩展成自己的测试。

## 学习路线

| Stage | 主题 | 亮点 |
| --- | --- | --- |
| `stage01_intro` | 基础入门 | 快速上手、状态图、工具调用链 |
| `stage02_agent_flows` | 构建智能体 | 反应式循环、工具路由 |
| `stage03_memory_systems` | 记忆策略 | 检查点、TrustCall、SQLite 持久化 |
| `stage04_state_management` | 状态建模 | TypedDict vs dataclass vs Pydantic、reducers |
| `stage05_conversation_control` | 对话控制 | Reducer 过滤、选择性提示、裁剪消息 |
| `stage06_streaming_and_monitoring` | 实时监控 | 流式模式、总结、Token 监控 |
| `stage07_debugging_and_iteration` | 调试迭代 | 断点、动态中断、时间回溯 |
| `stage08_production_ready` | 生产实践 | 并行检索、上下文综合 |

每个 Python 文件开头都有 “What You'll Learn / Lesson Flow” 的文档字符串，运行前即可快速了解内容。

## 快速开始

我们使用 [uv](https://docs.astral.sh/uv/) 管理依赖；如有需要，仍然可以导出传统的 `requirements.txt`。

```bash
git clone https://github.com/leslieo2/LangGraph-Mastery-Playbook
cd LangGraph-Mastery-Playbook
uv venv                    # 创建虚拟环境
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install .
```

> 需要锁定依赖？执行 `uv pip compile requirements.in -o requirements.txt` 即可生成。

### 环境变量

根据要运行的课程设置需要的 API Key：

```bash
export OPENAI_API_KEY="sk-..."        # 所有 LLM 演示都需要
export TAVILY_API_KEY="tvly-..."      # Stage 08 并行检索示例需要
export LANGSMITH_API_KEY="ls-..."     # 可选，启用支持课程的链路追踪
```

### LLM 模型 / 供应商配置

所有课程都通过 `src.langgraph_learning.utils.create_llm` 来构造聊天模型，该函数支持多家兼容 OpenAI 协议的供应商。默认使用 OpenAI 的 `gpt-5-nano`，也可以通过以下环境变量切换：

- `LLM_PROVIDER`：可选 `openai`（默认）、`openrouter`、`deepseek`。
- `LLM_MODEL`：覆盖默认模型名称，例如 `gpt-4o-mini` 或 `openrouter/anthropic/claude-3-haiku`。
- `LLM_TEMPERATURE`：可选的浮点数，覆盖采样温度。
- `LLM_API_KEY`：任何供应商的通用兜底密钥（若未设置对应供应商的专属变量）。
- 供应商专属密钥 / Base URL：
  - OpenRouter：`OPENROUTER_API_KEY`，可选 `OPENROUTER_BASE_URL`（默认 `https://openrouter.ai/api/v1`）。
  - DeepSeek：`DEEPSEEK_API_KEY`，可选 `DEEPSEEK_BASE_URL`（默认 `https://api.deepseek.com/v1`）。
  - OpenAI：`OPENAI_API_KEY`，可选 `OPENAI_BASE_URL`。

例如切换到 OpenRouter：

```bash
export LLM_PROVIDER="openrouter"
export OPENROUTER_API_KEY="or-..."
export LLM_MODEL="openrouter/anthropic/claude-3-haiku"
```

### 运行课程脚本

每个脚本都可以通过 `python -m` 运行（使用 uv 时也可用 `uv run ...`）：

```bash
# Stage 01 示例
python -m src.langgraph_learning.stage01_intro.quickstart
python -m src.langgraph_learning.stage01_intro.tool_calling_chain

# Stage 03 记忆系统
python -m src.langgraph_learning.stage03_memory_systems.agent_with_memory
python -m sr.langgraph_learning.stage03_memory_systems.trustcall_memory_agent

# Stage 07 调试流程
python -m src.langgraph_learning.stage07_debugging_and_iteration.breakpoints
```

多数课程会在对应模块的 `artifacts/` 目录下生成图形化的 PNG；流式课程会输出增量信息；调试课程可能会提示手动确认。


## 与 Notebook 教程的差异

- **可复现性。** 不依赖 Notebook 隐式状态——每次运行都从干净的 `main()` 开始。
- **更适合版本管理。** Diff 清晰，课程梗概就写在文档字符串里，代码旁边就有叙述。
- **易于集成。** 可以直接丢进 CI、用 pytest 包装、或者改造成自己的 LangGraph 服务模板。

如果你更喜欢 Notebook，也可以根据这些脚本改写，但本项目有意强调生产式的工作流程。

## 参与贡献

- Fork 仓库，为每个功能或课程改进创建独立分支，再提交 PR。
- 提交之前运行 `black .`，保持代码风格一致。
- 在提交 PR 前确保 `python -m compileall src` 可以通过。
- 新增课程时记得添加文档字符串概要，以便保持阶段化的学习体验。

玩转agent，尽情享受！🎯

## 致谢

- 本项目的结构和许多课程思路，灵感来自出色的 [Intro to LangGraph](https://academy.langchain.com/courses/take/intro-to-langgraph) 课程。
- 官方的 [LangChain + LangGraph 文档](https://docs.langchain.com/) 是跟随脚本学习时不可或缺的参考资料。
