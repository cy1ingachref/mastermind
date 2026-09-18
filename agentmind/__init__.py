"""AgentMind — multi-agent orchestration with mastermind."""
from __future__ import annotations

from .types import Task, Mission
from .orchestrator import Orchestrator

__version__ = "0.1.0"
__all__ = ["Task", "Mission", "Orchestrator"]
