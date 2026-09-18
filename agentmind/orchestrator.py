"""Orchestrator — task decomposition, delegation, and aggregation."""
from __future__ import annotations

import json
import time
from typing import Any

import requests
from rich.console import Console

from .providers import get_provider, BaseProvider
from .types import Task, Mission

console = Console()


class Orchestrator:
    """Decomposes missions into tasks and delegates to agents."""

    def __init__(self, mastermind: str, agents: list[str]):
        self.mastermind_name = mastermind
        self.agent_names = agents
        self.providers: dict[str, BaseProvider] = {}

        # Initialize providers
        try:
            self.providers[mastermind] = get_provider(mastermind)
        except Exception:
            pass

        for agent in agents:
            try:
                self.providers[agent] = get_provider(agent)
            except Exception:
                pass

    def _complete(self, provider_name: str, prompt: str, system: str | None = None) -> str:
        """Get completion from a provider."""
        provider = self.providers.get(provider_name)
        if not provider:
            return f"[ERROR: {provider_name} not available]"
        try:
            return provider.complete(prompt, system=system)
        except Exception as e:
            return f"[ERROR: {e}]"

    def plan(self, mission: Mission) -> list[Task]:
        """Mastermind decomposes the mission into tasks."""
        available = [self.mastermind_name] + self.agent_names
        agents_str = ", ".join(available)

        system = f"""You are the mastermind AI orchestrator. You decompose complex goals into atomic subtasks and assign each to the best-suited agent.

Available agents: {agents_str}
- {self.mastermind_name}: Mastermind (you) — complex reasoning, planning, final synthesis
- Other agents: Workers — specialized execution

Respond with JSON array:
[{{"description": "subtask description", "agent": "agent_name", "reason": "why this agent"}}]

Rules:
1. Break goal into 3-7 focused subtasks
2. Assign complex reasoning/analysis to mastermind
3. Assign specialized work to specific agents
4. Order matters — dependencies first
5. Keep descriptions atomic and actionable"""

        prompt = f"Decompose this mission into subtasks:\n\n{mission.goal}\n\nAvailable agents: {agents_str}"

        result = self._complete(self.mastermind_name, prompt, system)

        # Parse JSON response
        tasks = self._parse_plan(result, available)
        return tasks

    def _parse_plan(self, raw: str, available_agents: list[str]) -> list[Task]:
        """Parse plan from LLM output."""
        tasks = []
        # Try to extract JSON array
        try:
            # Find JSON in response
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start >= 0 and end > start:
                data = json.loads(raw[start:end])
                for i, item in enumerate(data):
                    agent = item.get("agent", self.mastermind_name)
                    if agent not in available_agents:
                        agent = self.mastermind_name
                    tasks.append(Task(
                        id=f"task_{i}",
                        description=item.get("description", ""),
                        agent=agent,
                    ))
        except (json.JSONDecodeError, Exception):
            pass

        # Fallback: create a single task for the mastermind
        if not tasks:
            tasks.append(Task(
                id="task_0",
                description=f"Complete the mission: {raw[:100]}",
                agent=self.mastermind_name,
            ))

        return tasks

    def execute(self, task: Task, mission: Mission) -> Task:
        """Execute a task using the assigned agent."""
        task.status = "running"
        task.started_at = time.time()

        provider_name = task.agent or self.mastermind_name

        system = f"""You are {provider_name}, a specialized AI agent working as part of a multi-agent team.

Mission goal: {mission.goal}

Your role: Execute the assigned task thoroughly and report results clearly."""

        # Include context from completed sibling tasks
        context_str = ""
        completed = [t for t in mission.tasks if t.status == "done" and t.id != task.id]
        if completed:
            context_str = "\n\nContext from completed tasks:\n"
            for t in completed:
                context_str += f"- [{t.agent}] {t.description}: {t.result[:200]}\n"

        prompt = f"Task: {task.description}\n{context_str}\n\nProvide a complete, actionable result."

        result = self._complete(provider_name, prompt, system)
        task.result = result
        task.status = "done" if not result.startswith("[ERROR") else "failed"
        task.completed_at = time.time()

        return task

    def aggregate(self, mission: Mission) -> str:
        """Mastermind aggregates all task results."""
        results_str = ""
        for task in mission.tasks:
            status_icon = "✓" if task.status == "done" else "✗"
            results_str += f"\n## [{status_icon}] {task.agent}: {task.description}\n{task.result}\n"

        system = """You are the mastermind AI. Synthesize all agent outputs into a coherent final deliverable. Resolve conflicts, fill gaps, and ensure quality."""

        prompt = f"""Mission: {mission.goal}

Results from all agents:
{results_str}

Synthesize a complete, coherent final deliverable."""

        return self._complete(self.mastermind_name, prompt, system)
