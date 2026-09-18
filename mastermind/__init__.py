"""MasterMind — multi-agent orchestration with mastermind."""
from __future__ import annotations

from .types import Task, Mission, TaskStatus, AgentConfig
from .orchestrator import Orchestrator

__version__ = "0.5.0"
__all__ = ["Task", "Mission", "TaskStatus", "AgentConfig", "Orchestrator"]
