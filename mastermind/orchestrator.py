"""Orchestrator — task decomposition, delegation, and aggregation."""
from __future__ import annotations

import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from rich.console import Console
from rich.table import Table

from .providers import get_provider, retry_with_backoff
from .types import Task, Mission, TaskStatus, AgentConfig

console = Console()


class Orchestrator:
    """Decomposes missions into tasks and delegates to agents with dependencies."""

    def __init__(self, mastermind: str, agents: list[str], one_mind=None):
        self.mastermind_name = mastermind
        self.agent_names = agents
        self.one_mind = one_mind  # OneMind instance for memory
        self.providers: dict[str, Any] = {}

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
        """Get completion from a provider with retry."""
        provider = self.providers.get(provider_name)
        if not provider:
            return f"[ERROR: {provider_name} not available]"
        try:
            return retry_with_backoff(
                lambda: provider.complete(prompt, system=system),
                max_retries=2,
                base_delay=1.0,
            )
        except Exception as e:
            return f"[ERROR: {e}]"

    def plan(self, mission: Mission) -> list[Task]:
        """Mastermind decomposes the mission into tasks with dependencies."""
        available = [self.mastermind_name] + self.agent_names
        agents_str = ", ".join(available)

        system = f"""You are the mastermind AI orchestrator. You decompose complex goals into atomic subtasks with dependencies.

Available agents: {agents_str}
- {self.mastermind_name}: Mastermind (you) — complex reasoning, planning, final synthesis
- Other agents: Workers — specialized execution

Respond with JSON array of tasks:
[{{
  "id": "task_0",
  "description": "subtask description",
  "agent": "agent_name",
  "depends_on": [],
  "reason": "why this agent and why no dependencies"
}}, ...]

Rules:
1. Break goal into 3-7 focused subtasks
2. Assign complex reasoning/analysis to mastermind
3. Assign specialized work to specific agents
4. Use depends_on to specify which tasks must complete before others
5. Root tasks (no dependencies) can run in parallel
6. Keep descriptions atomic and actionable
7. Each task id must be unique (task_0, task_1, etc.)"""

        prompt = f"Decompose this mission into subtasks with dependencies:\n\n{mission.goal}\n\nAvailable agents: {agents_str}"

        result = self._complete(self.mastermind_name, prompt, system)

        # Parse JSON response
        tasks = self._parse_plan(result, available)

        # Assign tasks to mission
        mission.tasks = tasks

        # Store mission plan in OneMind
        if self.one_mind:
            self.one_mind.remember(
                content=f"Mission: {mission.goal}",
                tags=["mastermind", "mission", "plan"],
                scope=f"mastermind/mission/{mission.id}",
                agent_id="mastermind",
            )

        return tasks

    def _parse_plan(self, raw: str, available_agents: list[str]) -> list[Task]:
        """Parse plan from LLM output."""
        tasks = []
        try:
            # Find JSON array in response
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start >= 0 and end > start:
                data = json.loads(raw[start:end])
                for item in data:
                    agent = item.get("agent", self.mastermind_name)
                    if agent not in available_agents:
                        agent = self.mastermind_name
                    tasks.append(Task(
                        id=item.get("id", f"task_{len(tasks)}"),
                        description=item.get("description", ""),
                        agent=agent,
                        depends_on=item.get("depends_on", []),
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

    def execute_parallel(self, mission: Mission, max_workers: int = 4, stream: bool = True) -> None:
        """Execute tasks in parallel respecting dependencies."""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while True:
                # Get ready tasks
                ready = mission.get_ready_tasks()
                if not ready:
                    # Check if all tasks are done or failed
                    all_done = all(
                        t.status in (TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.CANCELLED)
                        for t in mission.tasks
                    )
                    if all_done:
                        break
                    # Wait for running tasks to complete
                    time.sleep(0.5)
                    continue

                # Submit ready tasks
                futures = {}
                for task in ready:
                    future = executor.submit(self._execute_task, task, mission)
                    futures[future] = task
                    task.status = TaskStatus.RUNNING
                    task.started_at = time.time()

                    if stream:
                        console.print(f"  [blue]●[/blue] Starting: {task.agent} — {task.description[:50]}...")

                # Wait for submitted tasks
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        task.status = TaskStatus.FAILED
                        task.error = str(e)
                        task.completed_at = time.time()

    def _execute_task(self, task: Task, mission: Mission) -> Task:
        """Execute a single task using the assigned agent."""
        provider_name = task.agent or self.mastermind_name

        system = f"""You are {provider_name}, a specialized AI agent working as part of a multi-agent team.

Mission goal: {mission.goal}

Your role: Execute the assigned task thoroughly and report results clearly."""

        # Include context from completed sibling tasks
        context_str = ""
        completed = [t for t in mission.tasks if t.status == TaskStatus.DONE and t.id != task.id]
        if completed:
            context_str = "\n\nContext from completed tasks:\n"
            for t in completed:
                context_str += f"- [{t.agent}] {t.description}: {t.result[:300]}\n"

        prompt = f"Task: {task.description}\n{context_str}\n\nProvide a complete, actionable result."

        try:
            result = self._complete(provider_name, prompt, system)
            task.result = result
            task.status = TaskStatus.DONE if not result.startswith("[ERROR") else TaskStatus.FAILED
            task.completed_at = time.time()
            task.retry_count += 1

            # Store result in OneMind
            if self.one_mind and task.status == TaskStatus.DONE:
                self.one_mind.remember(
                    content=f"Task '{task.description}': {result[:500]}",
                    tags=["mastermind", "task", "result"],
                    scope=f"mastermind/mission/{mission.id}",
                    agent_id=task.agent,
                )

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = time.time()

        return task

    def aggregate(self, mission: Mission) -> str:
        """Mastermind aggregates all task results."""
        results_str = ""
        for task in mission.tasks:
            status_icon = "✓" if task.status == TaskStatus.DONE else "✗"
            results_str += f"\n## [{status_icon}] {task.agent}: {task.description}\n{task.result}\n"

        system = """You are the mastermind AI. Synthesize all agent outputs into a coherent final deliverable. Resolve conflicts, fill gaps, and ensure quality."""

        prompt = f"""Mission: {mission.goal}

Results from all agents:
{results_str}

Synthesize a complete, coherent final deliverable."""

        final = self._complete(self.mastermind_name, prompt, system)

        # Store final result in OneMind
        if self.one_mind:
            self.one_mind.remember(
                content=f"Final synthesis: {final[:1000]}",
                tags=["mastermind", "mission", "final"],
                scope=f"mastermind/mission/{mission.id}",
                agent_id="mastermind",
            )

        return final

    def get_cost_summary(self, mission: Mission) -> dict[str, Any]:
        """Get cost summary for the mission."""
        total_cost = sum(t.cost for t in mission.tasks)
        agent_costs: dict[str, float] = {}
        for t in mission.tasks:
            agent_costs[t.agent] = agent_costs.get(t.agent, 0.0) + t.cost
        return {
            "total": total_cost,
            "by_agent": agent_costs,
            "task_count": len(mission.tasks),
        }
