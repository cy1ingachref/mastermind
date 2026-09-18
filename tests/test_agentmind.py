"""Tests for AgentMind."""
from __future__ import annotations

import pytest
import os

from agentmind import Task, Mission
from agentmind.types import Task as Task2, Mission as Mission2


class TestTypes:
    def test_task_duration_calculation(self):
        import time
        task = Task(id="test", description="test task")
        task.started_at = time.time()
        time.sleep(0.01)
        task.completed_at = time.time()
        assert task.duration > 0

    def test_task_default_values(self):
        task = Task(id="test", description="test")
        assert task.status == "pending"
        assert task.agent == ""
        assert task.result == ""

    def test_mission_defaults(self):
        mission = Mission(id="m1", goal="test goal", mastermind="claude")
        assert mission.status == "planning"
        assert mission.tasks == []
        assert mission.agents == []


class TestOrchestrator:
    """Test orchestrator with mock providers."""

    def test_plan_parses_json(self):
        """Test plan parsing with valid JSON."""
        from agentmind.orchestrator import Orchestrator
        from unittest.mock import MagicMock, patch

        # Mock providers
        mock_mastermind = MagicMock()
        mock_mastermind.complete.return_value = '[{"description": "Design API", "agent": "claude"}, {"description": "Implement endpoints", "agent": "gpt"}]'

        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.mastermind_name = "claude"
        orchestrator.agent_names = ["gpt"]
        orchestrator.providers = {"claude": mock_mastermind}

        mission = Mission(id="test", goal="Build API", mastermind="claude", agents=["gpt"])
        tasks = orchestrator.plan(mission)

        assert len(tasks) == 2
        assert tasks[0].description == "Design API"
        assert tasks[0].agent == "claude"

    def test_plan_handles_invalid_json(self):
        """Test plan parsing falls back gracefully."""
        from agentmind.orchestrator import Orchestrator
        from unittest.mock import MagicMock

        mock_mastermind = MagicMock()
        mock_mastermind.complete.return_value = "Just do everything yourself"

        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.mastermind_name = "claude"
        orchestrator.agent_names = []
        orchestrator.providers = {"claude": mock_mastermind}

        mission = Mission(id="test", goal="Test", mastermind="claude")
        tasks = orchestrator.plan(mission)

        assert len(tasks) == 1  # Fallback task
        assert tasks[0].agent == "claude"

    def test_execute_task_success(self):
        """Test task execution."""
        from agentmind.orchestrator import Orchestrator
        from unittest.mock import MagicMock

        mock_claude = MagicMock()
        mock_claude.complete.return_value = "Task completed successfully"

        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.mastermind_name = "claude"
        orchestrator.agent_names = []
        orchestrator.providers = {"claude": mock_claude}

        mission = Mission(id="test", goal="Test goal", mastermind="claude")
        task = Task(id="task_0", description="Do something", agent="claude")

        result = orchestrator.execute(task, mission)

        assert result.status == "done"
        assert "successfully" in result.result

    def test_aggregate_results(self):
        """Test aggregation of results."""
        from agentmind.orchestrator import Orchestrator
        from unittest.mock import MagicMock

        mock_claude = MagicMock()
        mock_claude.complete.return_value = "Final synthesized result"

        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.mastermind_name = "claude"
        orchestrator.agent_names = []
        orchestrator.providers = {"claude": mock_claude}

        mission = Mission(id="test", goal="Test", mastermind="claude")
        mission.tasks = [
            Task(id="t1", description="T1", agent="claude", status="done", result="R1"),
            Task(id="t2", description="T2", agent="gpt", status="done", result="R2"),
        ]

        result = orchestrator.aggregate(mission)
        assert result == "Final synthesized result"


class TestProviderDetection:
    def test_list_providers(self):
        from agentmind.providers import list_providers
        providers = list_providers()
        assert "claude" in providers
        assert "gpt" in providers
        assert "gemini" in providers

    def test_list_available_with_no_keys(self):
        from agentmind.providers import list_available_providers
        # Save current keys
        saved = {}
        for var in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY",
                     "MOONSHOT_API_KEY", "XAI_API_KEY", "MISTRAL_API_KEY"]:
            saved[var] = os.environ.pop(var, None)

        try:
            available = list_available_providers()
            # Without keys, should return empty
            assert len(available) == 0
        finally:
            # Restore keys
            for var, val in saved.items():
                if val:
                    os.environ[var] = val


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
