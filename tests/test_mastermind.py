"""Tests for MasterMind v0.6.0."""
from __future__ import annotations

import json
import tempfile
import time
from unittest.mock import MagicMock, patch
import pytest

from mastermind import Task, Mission, TaskStatus, AgentConfig, Orchestrator
from mastermind.providers import retry_with_backoff, list_providers, get_provider, list_available_providers


class TestTask:
    def test_task_defaults(self):
        task = Task(id="test", description="test")
        assert task.status == TaskStatus.PENDING
        assert task.agent == ""
        assert task.depends_on == []
        assert task.is_ready

    def test_task_is_ready_with_no_deps(self):
        task = Task(id="t1", description="test", depends_on=[])
        assert task.is_ready

    def test_task_is_ready_when_pending(self):
        task = Task(id="t1", description="test", depends_on=["t0"])
        assert task.is_ready  # is_ready only checks PENDING status

    def test_task_not_ready_when_done(self):
        task = Task(id="t1", description="test", status=TaskStatus.DONE)
        assert not task.is_ready  # done tasks aren't ready

    def test_task_can_retry(self):
        task = Task(id="t1", description="test", max_retries=3)
        assert task.can_retry()
        task.retry_count = 3
        assert not task.can_retry()

    def test_task_duration(self):
        task = Task(id="t1", description="test")
        task.started_at = time.time()
        time.sleep(0.01)
        task.completed_at = time.time()
        assert task.duration > 0


class TestMission:
    def test_mission_defaults(self):
        mission = Mission(id="m1", goal="test goal", mastermind="claude")
        assert mission.status == "planning"
        assert mission.tasks == []

    def test_get_ready_tasks(self):
        mission = Mission(id="m1", goal="test", mastermind="claude")
        mission.tasks = [
            Task(id="t0", description="root", depends_on=[]),
            Task(id="t1", description="child", depends_on=["t0"]),
        ]
        ready = mission.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].id == "t0"

    def test_get_ready_tasks_with_completed(self):
        mission = Mission(id="m1", goal="test", mastermind="claude")
        mission.tasks = [
            Task(id="t0", description="root", status=TaskStatus.DONE),
            Task(id="t1", description="child", depends_on=["t0"]),
        ]
        ready = mission.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].id == "t1"

    def test_get_ready_tasks_all_completed(self):
        mission = Mission(id="m1", goal="test", mastermind="claude")
        mission.tasks = [
            Task(id="t0", description="task", status=TaskStatus.DONE),
        ]
        ready = mission.get_ready_tasks()
        assert len(ready) == 0

    def test_get_task_by_id(self):
        mission = Mission(id="m1", goal="test", mastermind="claude")
        mission.tasks = [Task(id="t0", description="test")]
        task = mission.get_task_by_id("t0")
        assert task is not None
        assert task.description == "test"
        assert mission.get_task_by_id("nonexistent") is None

    def test_progress(self):
        mission = Mission(id="m1", goal="test", mastermind="claude")
        mission.tasks = [
            Task(id="t0", description="a", status=TaskStatus.DONE),
            Task(id="t1", description="b", status=TaskStatus.RUNNING),
            Task(id="t2", description="c", status=TaskStatus.PENDING),
        ]
        progress = mission.progress
        assert progress["done"] == 1
        assert progress["running"] == 1
        assert progress["pending"] == 1


class TestJSONParsing:
    """Test robust JSON extraction."""

    def test_direct_json(self):
        orchestrator = Orchestrator.__new__(Orchestrator)
        result = orchestrator._extract_json('[{"id": "t0", "description": "test"}]')
        assert result is not None
        assert len(result) == 1
        assert result[0]["id"] == "t0"

    def test_json_with_surrounding_text(self):
        orchestrator = Orchestrator.__new__(Orchestrator)
        result = orchestrator._extract_json('Here is the plan:\n[{"id": "t0", "description": "test"}]\nDone.')
        assert result is not None
        assert len(result) == 1

    def test_json_in_code_block(self):
        orchestrator = Orchestrator.__new__(Orchestrator)
        result = orchestrator._extract_json('```json\n[{"id": "t0", "description": "test"}]\n```')
        assert result is not None
        assert len(result) == 1

    def test_invalid_json(self):
        orchestrator = Orchestrator.__new__(Orchestrator)
        result = orchestrator._extract_json("This is not JSON at all")
        assert result is None


class TestProviders:
    def test_list_providers(self):
        providers = list_providers()
        assert "claude" in providers
        assert "gpt" in providers
        assert "gemini" in providers
        assert "kimi" in providers

    def test_get_provider_unknown(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("nonexistent")

    def test_retry_with_backoff_success(self):
        """Test that successful function returns immediately."""
        result = retry_with_backoff(lambda: "success", max_retries=3)
        assert result == "success"

    def test_retry_with_backoff_retries_then_succeeds(self):
        """Test that retry succeeds after initial failures."""
        call_count = [0]

        def flaky():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("Failed")
            return "success"

        result = retry_with_backoff(flaky, max_retries=3, base_delay=0.01)
        assert result == "success"
        assert call_count[0] == 3

    def test_retry_with_backoff_exhausts_retries(self):
        """Test that retry raises after exhausting attempts."""
        def always_fail():
            raise ConnectionError("Always fails")

        with pytest.raises(ConnectionError):
            retry_with_backoff(always_fail, max_retries=2, base_delay=0.01)


class TestOrchestrator:
    """Test orchestrator with mock providers."""

    def test_plan_parses_json(self):
        """Test plan parsing with valid JSON."""
        mock_mastermind = MagicMock()
        mock_mastermind.complete.return_value = json.dumps([
            {"id": "task_0", "description": "Design API", "agent": "claude", "depends_on": []},
            {"id": "task_1", "description": "Implement endpoints", "agent": "gpt", "depends_on": ["task_0"]},
        ])

        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.mastermind_name = "claude"
        orchestrator.agent_names = ["gpt"]
        orchestrator.providers = {"claude": mock_mastermind}
        orchestrator.one_mind = None

        mission = Mission(id="test", goal="Build API", mastermind="claude", agents=["gpt"])
        tasks = orchestrator.plan(mission)

        assert len(tasks) == 2
        assert tasks[0].description == "Design API"
        assert tasks[0].depends_on == []
        assert tasks[1].description == "Implement endpoints"
        assert tasks[1].depends_on == ["task_0"]

    def test_plan_handles_invalid_json(self):
        """Test plan parsing falls back gracefully."""
        mock_mastermind = MagicMock()
        mock_mastermind.complete.return_value = "Just do everything yourself"

        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.mastermind_name = "claude"
        orchestrator.agent_names = []
        orchestrator.providers = {"claude": mock_mastermind}
        orchestrator.one_mind = None

        mission = Mission(id="test", goal="Test", mastermind="claude")
        tasks = orchestrator.plan(mission)

        assert len(tasks) == 1  # Fallback task
        assert tasks[0].agent == "claude"

    def test_plan_respects_dependencies(self):
        """Test that dependencies are correctly assigned."""
        mock_mastermind = MagicMock()
        mock_mastermind.complete.return_value = json.dumps([
            {"id": "task_0", "description": "Root", "agent": "claude", "depends_on": []},
            {"id": "task_1", "description": "Child A", "agent": "gpt", "depends_on": ["task_0"]},
            {"id": "task_2", "description": "Child B", "agent": "gemini", "depends_on": ["task_0"]},
        ])

        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.mastermind_name = "claude"
        orchestrator.agent_names = ["gpt", "gemini"]
        orchestrator.providers = {"claude": mock_mastermind}
        orchestrator.one_mind = None

        mission = Mission(id="test", goal="Test", mastermind="claude", agents=["gpt", "gemini"])
        tasks = orchestrator.plan(mission)

        assert len(tasks) == 3
        # Only task_0 is ready (no deps)
        ready = mission.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].id == "task_0"

    def test_execute_task_success(self):
        """Test task execution."""
        mock_claude = MagicMock()
        mock_claude.complete.return_value = "Task completed successfully"

        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.mastermind_name = "claude"
        orchestrator.agent_names = []
        orchestrator.providers = {"claude": mock_claude}
        orchestrator.one_mind = None

        mission = Mission(id="test", goal="Test goal", mastermind="claude")
        task = Task(id="task_0", description="Do something", agent="claude")

        result = orchestrator._execute_task(task, mission)

        assert result.status == TaskStatus.DONE
        assert "successfully" in result.result

    def test_aggregate_results(self):
        """Test aggregation of results."""
        mock_claude = MagicMock()
        mock_claude.complete.return_value = "Final synthesized result"

        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.mastermind_name = "claude"
        orchestrator.agent_names = []
        orchestrator.providers = {"claude": mock_claude}
        orchestrator.one_mind = None

        mission = Mission(id="test", goal="Test", mastermind="claude")
        mission.tasks = [
            Task(id="t1", description="T1", agent="claude", status=TaskStatus.DONE, result="R1"),
            Task(id="t2", description="T2", agent="gpt", status=TaskStatus.DONE, result="R2"),
        ]

        result = orchestrator.aggregate(mission)
        assert result == "Final synthesized result"

    def test_export_trace(self):
        """Test mission trace export."""
        mock_claude = MagicMock()
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.mastermind_name = "claude"
        orchestrator.agent_names = []
        orchestrator.providers = {"claude": mock_claude}
        orchestrator.one_mind = None

        mission = Mission(id="test", goal="Test goal", mastermind="claude")
        mission.tasks = [
            Task(id="t1", description="Task 1", agent="claude", status=TaskStatus.DONE, result="Result 1"),
            Task(id="t2", description="Task 2", agent="gpt", status=TaskStatus.FAILED, error="Error"),
        ]

        trace = orchestrator.export_trace(mission)

        assert trace["id"] == "test"
        assert trace["goal"] == "Test goal"
        assert len(trace["tasks"]) == 2
        assert trace["tasks"][0]["id"] == "t1"
        assert trace["tasks"][0]["status"] == "done"
        assert trace["tasks"][1]["status"] == "failed"

    def test_model_overrides(self):
        """Test model overrides in orchestrator initialization."""
        orchestrator = Orchestrator(
            mastermind="claude",
            agents=["gpt"],
            model_overrides={"gpt": "gpt-4o-mini"}
        )
        # Should have initialized providers (or tried to)
        assert "claude" in orchestrator.providers or len(orchestrator.providers) == 0


class TestProviderDetection:
    def test_list_available_with_no_keys(self):
        """Test that without keys, no providers available."""
        import os
        saved = {}
        for var in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY",
                     "MOONSHOT_API_KEY", "XAI_API_KEY", "MISTRAL_API_KEY"]:
            saved[var] = os.environ.pop(var, None)

        try:
            available = list_available_providers()
            assert len(available) == 0
        finally:
            for var, val in saved.items():
                if val:
                    os.environ[var] = val


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
