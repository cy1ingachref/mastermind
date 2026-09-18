"""MasterMind CLI — multi-agent orchestration."""
from __future__ import annotations

import json
import os

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .types import Task, Mission, TaskStatus, AgentConfig
from .orchestrator import Orchestrator
from .providers import list_providers, list_available_providers

console = Console()


def _run_mission(
    goal: str,
    mastermind: str,
    agents: list[str],
    use_memory: bool = False,
    max_workers: int = 4,
    model_overrides: dict[str, str] | None = None,
    export_path: str | None = None,
) -> tuple[Mission, str, float]:
    """Run a mission (shared between CLI commands)."""
    # OneMind integration
    one_mind = None
    if use_memory:
        try:
            from onemind import OneMind
            one_mind = OneMind()
            console.print("[dim]OneMind memory enabled[/dim]")
        except ImportError:
            console.print("[yellow]OneMind not installed, running without memory[/yellow]")

    mission_id = __import__("hashlib").sha256(goal.encode()).hexdigest()[:8]
    mission = Mission(
        id=mission_id,
        goal=goal,
        mastermind=mastermind,
        agents=agents,
        created_at=__import__("time").time(),
    )

    orchestrator = Orchestrator(mastermind, agents, one_mind=one_mind, model_overrides=model_overrides)

    # Phase 1: Plan
    console.print(f"\n[bold cyan]Phase 1: Planning[/bold cyan]")
    mission.tasks = orchestrator.plan(mission)
    mission.status = "executing"
    console.print(f"[dim]Decomposed into {len(mission.tasks)} tasks[/dim]")

    # Show plan table
    table = Table(title="Mission Plan")
    table.add_column("#", style="dim", width=4)
    table.add_column("Agent", style="cyan")
    table.add_column("Task", style="green")
    table.add_column("Depends On", style="yellow")
    for i, task in enumerate(mission.tasks):
        deps = ", ".join(task.depends_on) if task.depends_on else "-"
        table.add_row(str(i + 1), task.agent, task.description[:50], deps)
    console.print(table)

    # Phase 2: Execute
    console.print(f"\n[bold cyan]Phase 2: Executing (parallel, {max_workers} workers)[/bold cyan]")
    orchestrator.execute_parallel(mission, max_workers=max_workers)

    # Phase 3: Aggregate
    console.print("\n[bold cyan]Phase 3: Aggregating[/bold cyan]")
    final = orchestrator.aggregate(mission)
    mission.status = "completed"
    mission.completed_at = __import__("time").time()

    # Cost summary
    cost_summary = orchestrator.get_cost_summary(mission)
    mission.total_cost = cost_summary["total"]

    # Export trace if requested
    if export_path:
        trace = orchestrator.export_trace(mission)
        with open(export_path, "w") as f:
            json.dump(trace, f, indent=2)
        console.print(f"[dim]Trace exported to {export_path}[/dim]")

    return mission, final, mission.total_cost


@click.group()
@click.version_option(version=__version__, prog_name="mastermind")
def cli():
    """MasterMind — multi-agent orchestration with mastermind.

    Pick a mastermind AI to decompose complex tasks,
    delegate to other agents in parallel, and aggregate results.
    """
    pass


@cli.command()
@click.argument("goal")
@click.option("--mastermind", "-m", default="claude", help="Mastermind agent (claude, gpt, gemini, kimi, grok, mistral)")
@click.option("--agents", "-a", default=None, help="Comma-separated worker agents")
@click.option("--memory/--no-memory", default=False, help="Enable OneMind for cross-mission memory")
@click.option("--workers", "-w", type=int, default=4, help="Max parallel workers (default: 4)")
@click.option("--model", "-M", multiple=True, help="Model override (format: agent:model, e.g., gpt:gpt-4o-mini)")
@click.option("--export", "-e", default=None, help="Export mission trace to JSON file")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed task output")
def run(goal: str, mastermind: str, agents: str | None, memory: bool, workers: int, model: tuple[str, ...], export: str | None, verbose: bool):
    """Run a multi-agent mission.

    Example:
        mastermind run "Build a REST API with auth and docs" -m claude -a gpt,gemini,kimi -w 3
    """
    agent_list = [a.strip() for a in agents.split(",")] if agents else []

    # Parse model overrides
    model_overrides = {}
    for m in model:
        if ":" in m:
            agent, model_name = m.split(":", 1)
            model_overrides[agent.strip()] = model_name.strip()

    mission, result, cost = _run_mission(goal, mastermind, agent_list, memory, workers, model_overrides, export)

    console.print(f"\n[bold green]Mission {mission.id} completed[/bold green]")
    console.print(f"[dim]Duration: {mission.completed_at - mission.created_at:.1f}s[/dim]")
    console.print(f"[dim]Cost: ${cost:.6f}[/dim]")

    # Show progress summary
    progress = mission.progress
    console.print(f"\n[dim]Tasks: {progress.get('done', 0)} done, {progress.get('failed', 0)} failed[/dim]")

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
@click.option("--model", "-M", multiple=True, help="Model override (format: agent:model)")
def plan(goal: str, mastermind: str, agents: str | None, model: tuple[str, ...]):
    """Plan a mission without executing (dry run)."""
    agent_list = [a.strip() for a in agents.split(",")] if agents else []

    # Parse model overrides
    model_overrides = {}
    for m in model:
        if ":" in m:
            agent, model_name = m.split(":", 1)
            model_overrides[agent.strip()] = model_name.strip()

    orchestrator = Orchestrator(mastermind, agent_list, model_overrides=model_overrides)
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
    table.add_column("Depends On", style="yellow")

    for i, task in enumerate(tasks):
        deps = ", ".join(task.depends_on) if task.depends_on else "-"
        table.add_row(str(i + 1), task.agent, task.description[:60], deps)

    console.print(table)


@cli.command()
def demo():
    """Run a demo mission (no API keys needed)."""
    console.print(Panel(
        "[bold]MasterMind Demo[/bold]\n\n"
        "This demo shows how tasks are decomposed.\n"
        "No API keys required.\n\n"
        "In real usage, the mastermind AI would:\n"
        "1. Decompose the goal into subtasks with dependencies\n"
        "2. Run independent tasks in parallel\n"
        "3. Store results in OneMind for memory across missions\n"
        "4. Aggregate all results into a final deliverable\n"
        "5. Track cost per agent and mission",
        title="Demo Mode",
        border_style="blue",
    ))

    goal = "Build a REST API for a todo app with authentication and documentation"

    tasks = [
        Task(id="task_0", description="Design API schema and endpoints", agent="mastermind", depends_on=[]),
        Task(id="task_1", description="Implement JWT authentication", agent="gpt", depends_on=["task_0"]),
        Task(id="task_2", description="Write API documentation", agent="gemini", depends_on=["task_0"]),
        Task(id="task_3", description="Write integration tests", agent="claude", depends_on=["task_1", "task_2"]),
    ]

    console.print(f"\n[bold]Goal:[/bold] {goal}\n")

    table = Table(title="Simulated Task Decomposition")
    table.add_column("#", style="dim", width=4)
    table.add_column("Agent", style="cyan")
    table.add_column("Task", style="green")
    table.add_column("Depends On", style="yellow")

    for i, task in enumerate(tasks):
        deps = ", ".join(task.depends_on) if task.depends_on else "-"
        table.add_row(str(i + 1), task.agent, task.description, deps)

    console.print(table)
    console.print("\n[dim]Run with real API keys to execute missions.[/dim]")


if __name__ == "__main__":
    cli()
