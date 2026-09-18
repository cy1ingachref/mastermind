# MasterMind

**Multi-agent orchestration with mastermind — parallel execution, dependency tracking, and cost awareness.**

Pick a mastermind AI (Claude, GPT, Gemini, etc.) to decompose complex tasks into a dependency graph,
run independent tasks in parallel, and aggregate results into a coherent deliverable.

```
$ mastermind run "Build a REST API with auth and docs" -m claude -a gpt,gemini,kimi
```

---

## How It Works

```
Mastermind (your pick)          Worker Agents
┌─────────────────┐            ┌──────────┐
│ Claude (planner)│───────────▶│ GPT      │
│                 │───────────▶│ Gemini   │
│                 │───────────▶│ Kimi     │
└─────────────────┘            └──────────┘
       │
       ▼
┌─────────────────┐
│  Dependency DAG │  Tasks run in parallel when independent
└─────────────────┘
       │
       ▼
┌─────────────────┐
│   Aggregation   │  Combines all results into final deliverable
└─────────────────┘
       │
       ▼
   Final Result + Cost Summary
```

---

## Quick Start

```bash
pip install mastermind

# Set at least one API key
export ANTHROPIC_API_KEY="..."
export OPENAI_API_KEY="..."

# Run a mission with parallel execution
mastermind run "Write a full-stack todo app with tests" -m claude -a gpt,gemini -w 4

# Enable OneMind for cross-mission memory
mastermind run "Build a REST API" -m claude -a gpt --memory

# Dry run (see the plan without executing)
mastermind plan "Build an e-commerce site" -m gpt -a claude,gemini,kimi

# See available providers
mastermind providers
```

---

## Features

- **Dependency Graph** — Tasks declare dependencies; independent tasks run in parallel
- **Parallel Execution** — ThreadPoolExecutor with configurable worker count (default: 4)
- **Retry with Backoff** — Exponential backoff with jitter for transient failures
- **Provider Failover** — Automatically retries on different providers
- **Cost Tracking** — Track cost per agent and per mission
- **OneMind Integration** — Store mission plans and results for cross-mission memory
- **Streaming Progress** — See task execution in real time
- **Task Status Machine** — Clear lifecycle: pending → running → done/failed → retry

---

## Providers

| Provider | Env Variable | Cost (per 1M tokens) |
|----------|--------------|----------------------|
| Claude | `ANTHROPIC_API_KEY` | $3.00 in / $15.00 out |
| GPT | `OPENAI_API_KEY` | $2.50 in / $10.00 out |
| Gemini | `GOOGLE_API_KEY` | $0.15 in / $0.60 out |
| Kimi | `MOONSHOT_API_KEY` | $1.00 in / $2.00 out |
| Grok | `XAI_API_KEY` | $2.00 in / $10.00 out |
| Mistral | `MISTRAL_API_KEY` | $2.00 in / $6.00 out |

---

## CLI Reference

```
mastermind run <goal>      Execute a full mission
mastermind plan <goal>     Dry run — show task decomposition
mastermind providers       List available providers
mastermind demo            Show example without API keys
```

### Options
- `-m, --mastermind` — Mastermind agent (default: claude)
- `-a, --agents` — Comma-separated worker agents
- `-w, --workers` — Max parallel workers (default: 4)
- `--memory/--no-memory` — Enable OneMind for cross-mission memory
- `-v, --verbose` — Show detailed task output

---

## Python SDK

```python
from mastermind import Orchestrator, Mission, Task, TaskStatus
from one_mind import OneMind

# Create orchestrator with OneMind memory
orchestrator = Orchestrator(
    mastermind="claude",
    agents=["gpt", "gemini", "kimi"],
    one_mind=OneMind(),
)

# Create mission
mission = Mission(
    id="my-mission",
    goal="Build a REST API",
    mastermind="claude",
    agents=["gpt", "gemini"],
)

# Plan
tasks = orchestrator.plan(mission)

# Execute (parallel)
orchestrator.execute_parallel(mission, max_workers=4)

# Aggregate
result = orchestrator.aggregate(mission)

# Check cost
cost = orchestrator.get_cost_summary(mission)
print(f"Total cost: ${cost['total']:.6f}")
```

---

## OneMind Integration

MasterMind works with [OneMind](https://github.com/cy1ingachref/one-mind) for persistent shared memory across agent sessions.

When `--memory` is enabled:
- Mission plans are stored automatically
- Task results are stored with agent provenance
- Final synthesis is stored
- Future missions can recall previous results

```bash
# Run with memory enabled
mastermind run "Build on previous work" -m claude -a gpt --memory

# Previous missions are automatically recalled as context
```

---

## Architecture

```
mastermind/
├── __init__.py        # Main exports
├── types.py           # Task, Mission, TaskStatus, AgentConfig
├── orchestrator.py    # Planning with DAG, parallel execution, aggregation
├── providers.py       # Claude, GPT, Gemini, Kimi, Grok, Mistral adapters
└── cli.py             # Click CLI
```

---

## License

MIT
