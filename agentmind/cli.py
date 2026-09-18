"""AgentMind CLI — multi-agent orchestration."""
from __future__ import annotations

import sys

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .types import Task, Mission
from .orchestrator import Orchestrator
from .providers import list_providers, list_available_providers

console = Console()


def _run_mission(goal: str, mastermind: str, agents: list[str]) -> tuple[Mission, str]:
    """Run a mission (shared between CLI commands)."""
    mission_id = __import__("hashlib").sha256(goal.encode()).hexdigest()[:8]
    mission = Mission(
        id=mission_id,
        goal=goal,
        mastermind=mastermind,
        agents=agents,
        created_at=__import__("time").time(),
    )

    orchestrator = Orchestrator(mastermind, agents)

    # Phase 1: Plan
    console.print("\n[bold cyan]Phase 1: Planning[/bold cyan]")
    mission.tasks = orchestrator.plan(mission)
    mission.status = "executing"
    console.print(f"[dim]Decomposed into {len(mission.tasks)} tasks[/dim]")

    # Phase 2: Execute
    console.print("\n[bold cyan]Phase 2: Executing[/bold cyan]")
    for task in mission.tasks:
        if task.agent:
            console.print(f"  [yellow]→[/yellow] {task.agent}: {task.description[:50]}...")
        else:
            console.print(f"  [blue]●[/blue] Mastermind: {task.description[:50]}...")

        result = orchestrator.execute(task, mission)
        if result.status == "done":
            console.print(f"  [green]✓[/green] Done ({result.duration:.1f}s)")
        else:
            console.print(f"  [red]✗[/red] Failed: {result.error}")

    # Phase 3: Aggregate
    console.print("\n[bold cyan]Phase 3: Aggregating[/bold cyan]")
    final = orchestrator.aggregate(mission)
    mission.status = "completed"
    mission.completed_at = __import__("time").time()

    return mission, final


@click.group()
@click.version_option(version=__version__, prog_name="agentmind")
def cli():
    """AgentMind — multi-agent orchestration with mastermind.

    Pick a mastermind AI to decompose complex tasks,
    delegate to other agents, and aggregate results.
    """
    pass


@cli.command()
@click.argument("goal")
@click.option("--mastermind", "-m", default="claude", help="Mastermind agent (claude, gpt, gemini, kimi, grok, mistral)")
@click.option("--agents", "-a", default=None, help="Comma-separated worker agents")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed task output")
def run(goal: str, mastermind: str, agents: str | None, verbose: bool):
    """Run a multi-agent mission.

    Example:
        agentmind run "Build a REST API with auth and docs" -m claude -a gpt,gemini,kimi
    """
    agent_list = [a.strip() for a in agents.split(",")] if agents else []

    mission, result = _run_mission(goal, mastermind, agent_list)

    console.print(f"\n[bold green]Mission {mission.id} completed[/bold green]")
    console.print("\n[bold]Final Result:[/bold]")
    console.print(Panel(result, title="Synthesis", border_style="green"))


@cli.command()
def providers():
    """List available AI providers."""
    available = list_available_providers()
    all_provs = list_providers()

    table = Table(title="AI Providers")
    table.add_column("Provider", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Key", style="yellow")

    for name in all_provs:
        status = "[green]✓ Available[/green]" if name in available else "[dim]✗ No key[/dim]"
        from .providers import PROVIDERS
        env_var = PROVIDERS[name].env_var
        table.add_row(name, status, env_var)

    console.print(table)
    console.print("\n[dim]Set API keys as environment variables to enable providers.[/dim]")


@cli.command()
@click.argument("goal")
@click.option("--mastermind", "-m", default="claude", help="Mastermind agent")
@click.option("--agents", "-a", default=None, help="Comma-separated worker agents")
def plan(goal: str, mastermind: str, agents: str | None):
    """Plan a mission without executing (dry run)."""
    agent_list = [a.strip() for a in agents.split(",")] if agents else []

    orchestrator = Orchestrator(mastermind, agent_list)
    mission_id = __import__("hashlib").sha256(goal.encode()).hexdigest()[:8]
    mission = Mission(
        id=mission_id,
        goal=goal,
        mastermind=mastermind,
        agents=agent_list,
        created_at=__import__("time").time(),
    )

    tasks = orchestrator.plan(mission)

    console.print(f"[bold]Mission Plan:[/bold] {goal}")
    console.print(f"[dim]Mastermind: {mastermind}, Workers: {', '.join(agent_list)}[/dim]\n")

    table = Table(title="Task Decomposition")
    table.add_column("#", style="dim", width=4)
    table.add_column("Agent", style="cyan")
    table.add_column("Task", style="green")

    for i, task in enumerate(tasks):
        table.add_row(str(i + 1), task.agent, task.description[:60])

    console.print(table)


@cli.command()
def demo():
    """Run a demo mission (no API keys needed)."""
    console.print(Panel(
        "[bold]AgentMind Demo[/bold]\n\n"
        "This demo shows how tasks are decomposed.\n"
        "No API keys required.\n\n"
        "In real usage, the mastermind AI would:\n"
        "1. Decompose the goal into subtasks\n"
        "2. Delegate each subtask to worker agents\n"
        "3. Aggregate all results into a final deliverable",
        title="Demo Mode",
        border_style="blue",
    ))

    goal = "Build a REST API for a todo app with authentication and documentation"

    tasks = [
        Task(id="task_0", description="Design API schema and endpoints", agent="mastermind"),
        Task(id="task_1", description="Implement JWT authentication", agent="gpt"),
        Task(id="task_2", description="Write API documentation", agent="gemini"),
        Task(id="task_3", description="Write integration tests", agent="claude"),
    ]

    console.print(f"\n[bold]Goal:[/bold] {goal}\n")

    table = Table(title="Simulated Task Decomposition")
    table.add_column("#", style="dim", width=4)
    table.add_column("Agent", style="cyan")
    table.add_column("Task", style="green")

    for i, task in enumerate(tasks):
        table.add_row(str(i + 1), task.agent, task.description)

    console.print(table)
    console.print("\n[dim]Run with real API keys to execute missions.[/dim]")


if __name__ == "__main__":
    cli()
