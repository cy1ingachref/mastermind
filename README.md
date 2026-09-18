# MasterMind

**Multi-agent orchestration with mastermind.**

Pick a mastermind AI (Claude, GPT, Gemini, etc.) to decompose complex tasks,
delegate to other agents, and aggregate results into a coherent deliverable.

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
│   Aggregation   │  ← Combines all results
└─────────────────┘
       │
       ▼
   Final Result
```

---

## Quick Start

```bash
pip install mastermind

# Set at least one API key
export ANTHROPIC_API_KEY="..."
export OPENAI_API_KEY="..."

# Run a mission
mastermind run "Write a full-stack todo app with tests" -m claude -a gpt,gemini

# Dry run (see the plan without executing)
mastermind plan "Build an e-commerce site" -m gpt -a claude,gemini,kimi

# See available providers
mastermind providers
```

---

## Providers

| Provider | Env Variable | Models |
|----------|--------------|--------|
| Claude | `ANTHROPIC_API_KEY` | claude-sonnet-4-20250514 |
| GPT | `OPENAI_API_KEY` | gpt-4o |
| Gemini | `GOOGLE_API_KEY` | gemini-2.5-flash |
| Kimi | `MOONSHOT_API_KEY` | moonshot-v1-8k |
| Grok | `XAI_API_KEY` | grok-2-latest |
| Mistral | `MISTRAL_API_KEY` | mistral-large-latest |

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

---

## OneMind Integration

MasterMind works with [OneMind](https://github.com/cy1ingachref/one-mind) for
persistent shared memory across agent sessions.

```python
from mastermind import Orchestrator
from one_mind import remember, recall

# Store mission results for future reference
remember("Auth module uses JWT with RS256", tags=["security"])

# Recall in a later mission
context = recall("authentication")
```

---

## Architecture

```
mastermind/
├── __init__.py        # Main exports
├── types.py           # Task, Mission dataclasses
├── orchestrator.py    # Planning, delegation, aggregation
├── providers.py       # Claude, GPT, Gemini, Kimi, Grok, Mistral adapters
└── cli.py             # Click CLI
```

---

## License

MIT
