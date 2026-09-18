"""AgentMind types — data models for missions and tasks."""
from __future__ import annotations

import hashlib
import time
from typing import Any
from dataclasses import dataclass, field


@dataclass
class Task:
    """A unit of work."""
    id: str
    description: str
    agent: str = ""
    status: str = "pending"  # pending, running, done, failed
    result: str = ""
    error: str = ""
    started_at: float = 0.0
    completed_at: float = 0.0
    subtasks: list["Task"] = field(default_factory=list)

    @property
    def duration(self) -> float:
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return 0.0


@dataclass
class Mission:
    """A complete mission with tasks."""
    id: str
    goal: str
    mastermind: str
    agents: list[str] = field(default_factory=list)
    tasks: list[Task] = field(default_factory=list)
    status: str = "planning"
    created_at: float = 0.0
    completed_at: float = 0.0
    context: dict[str, Any] = field(default_factory=dict)
