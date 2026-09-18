"""MasterMind types — data models for missions, tasks, and agents."""
from __future__ import annotations

import hashlib
import time
from enum import Enum
from typing import Any
from dataclasses import dataclass, field


class TaskStatus(str, Enum):
    """Task lifecycle states."""
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """A unit of work with dependencies."""
    id: str
    description: str
    agent: str = ""
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""
    error: str = ""
    started_at: float = 0.0
    completed_at: float = 0.0
    retry_count: int = 0
    max_retries: int = 3
    depends_on: list[str] = field(default_factory=list)
    cost: float = 0.0

    @property
    def duration(self) -> float:
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return 0.0

    @property
    def is_ready(self) -> bool:
        return self.status == TaskStatus.PENDING

    def can_retry(self) -> bool:
        return self.retry_count < self.max_retries


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    name: str
    provider: str
    model: str = ""
    system_prompt: str = ""
    max_tokens: int = 4096
    temperature: float = 0.3


@dataclass
class Mission:
    """A complete mission with tasks and metadata."""
    id: str
    goal: str
    mastermind: str
    agents: list[str] = field(default_factory=list)
    tasks: list[Task] = field(default_factory=list)
    status: str = "planning"
    created_at: float = 0.0
    completed_at: float = 0.0
    context: dict[str, Any] = field(default_factory=dict)
    total_cost: float = 0.0

    def get_ready_tasks(self) -> list[Task]:
        """Get tasks whose dependencies are all done."""
        done_ids = {t.id for t in self.tasks if t.status == TaskStatus.DONE}
        return [
            t for t in self.tasks
            if t.status == TaskStatus.PENDING
            and all(dep in done_ids for dep in t.depends_on)
        ]

    def get_task_by_id(self, task_id: str) -> Task | None:
        for t in self.tasks:
            if t.id == task_id:
                return t
        return None

    @property
    def progress(self) -> dict[str, int]:
        counts = {}
        for t in self.tasks:
            counts[t.status.value] = counts.get(t.status.value, 0) + 1
        return counts
