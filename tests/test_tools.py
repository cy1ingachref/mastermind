"""Tests for tool calling."""
from __future__ import annotations

import os
import json
import tempfile
import pytest

from mastermind.tools import (
    Tool, ToolResult, ToolParameter, ToolRegistry,
    AgentLoop, ToolCall, get_registry
)


class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        tool = Tool()
        tool.name = "test_tool"
        registry.register(tool)
        assert registry.get("test_tool") is not None
        assert registry.get("nonexistent") is None

    def test_execute_tool(self):
        registry = ToolRegistry()
        tool = Tool()
        tool.name = "test_tool"
        def execute(**kwargs):
            return ToolResult(success=True, output="test output")
        tool.execute = execute
        registry.register(tool)

        result = registry.execute("test_tool", {})
        assert result.success
        assert result.output == "test output"

    def test_execute_nonexistent(self):
        registry = ToolRegistry()
        result = registry.execute("nonexistent", {})
        assert not result.success
        assert "not found" in result.error

    def test_get_schemas(self):
        registry = ToolRegistry()
        tool = Tool()
        tool.name = "test"
        tool.description = "A test tool"
        tool.parameters = [ToolParameter(name="param1", type="string", description="A param")]
        registry.register(tool)

        schemas = registry.get_schemas()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "test"


class TestBuiltInTools:
    def test_file_read(self):
        from mastermind.tools.builtins import FileReadTool
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test content")
            f.flush()
            path = f.name
        try:
            tool = FileReadTool()
            result = tool.execute(path=path)
            assert result.success
            assert result.output == "test content"
        finally:
            os.unlink(path)

    def test_file_read_not_found(self):
        from mastermind.tools.builtins import FileReadTool
        tool = FileReadTool()
        result = tool.execute(path="/nonexistent/file.txt")
        assert not result.success

    def test_file_write(self):
        from mastermind.tools.builtins import FileWriteTool
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.txt")
            tool = FileWriteTool()
            result = tool.execute(path=path, content="written content")
            assert result.success
            with open(path) as f:
                assert f.read() == "written content"

    def test_file_list(self):
        from mastermind.tools.builtins import FileListTool
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some files
            open(os.path.join(tmpdir, "a.txt"), "w").close()
            open(os.path.join(tmpdir, "b.txt"), "w").close()
            tool = FileListTool()
            result = tool.execute(path=tmpdir, pattern="*")
            assert result.success
            assert "a.txt" in result.output
            assert "b.txt" in result.output

    def test_code_execute(self):
        from mastermind.tools.builtins import CodeExecuteTool
        tool = CodeExecuteTool()
        result = tool.execute(code='print("hello world")')
        assert result.success
        assert "hello world" in result.output

    def test_code_execute_timeout(self):
        from mastermind.tools.builtins import CodeExecuteTool
        tool = CodeExecuteTool()
        result = tool.execute(code='import time; time.sleep(5)', timeout=1)
        assert not result.success
        assert "Timeout" in result.error

    def test_task_complete(self):
        from mastermind.tools.builtins import TaskCompleteTool
        tool = TaskCompleteTool()
        result = tool.execute(result="Task done", summary="All good")
        assert result.success
        assert "Task complete" in result.output
        assert "Task done" in result.output


class TestAgentLoop:
    def test_agent_with_no_tools(self):
        """Test agent that doesn't use tools."""
        mock_provider = type('MockProvider', (), {
            'complete': lambda self, prompt, system=None, **kw: "Final answer: success"
        })()
        registry = ToolRegistry()
        loop = AgentLoop(mock_provider, registry, max_steps=5)
        result, trace = loop.run("Test task")
        assert result == "Final answer: success"
        assert len(trace) == 1
        assert trace[0]["type"] == "response"

    def test_agent_with_tool_call(self):
        """Test agent that calls a tool."""
        call_count = [0]
        def mock_complete(prompt, system=None, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                return '{"tool": "task_complete", "arguments": {"result": "done"}}'
            return "Final answer"

        mock_provider = type('MockProvider', (), {
            'complete': lambda self, prompt, system=None, **kw: mock_complete(prompt, system=system, **kw)
        })()
        registry = ToolRegistry()
        # Register task_complete
        from mastermind.tools.builtins import TaskCompleteTool
        registry.register(TaskCompleteTool())

        loop = AgentLoop(mock_provider, registry, max_steps=5)
        result, trace = loop.run("Test task")
        assert "done" in result
        assert len(trace) == 1
        assert trace[0]["type"] == "tool_call"

    def test_max_steps_reached(self):
        """Test that loop exits after max steps if no completion."""
        # Mock that always returns a non-completing tool call
        def mock_complete(prompt, system=None, **kw):
            return '{"tool": "file_read", "arguments": {"path": "/tmp/test"}}'

        mock_provider = type('MockProvider', (), {
            'complete': lambda self, prompt, system=None, **kw: mock_complete(prompt, system=system, **kw)
        })()
        registry = ToolRegistry()
        # Register file_read so it can execute but not complete
        from mastermind.tools.builtins import FileReadTool
        registry.register(FileReadTool())

        loop = AgentLoop(mock_provider, registry, max_steps=3)
        result, trace = loop.run("Test task")
        # Should have tried max_steps times
        assert len(trace) == 3


class TestGlobalRegistry:
    def test_builtins_registered(self):
        registry = get_registry()
        tools = [t.name for t in registry.list_tools()]
        assert "file_read" in tools
        assert "file_write" in tools
        assert "code_execute" in tools


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
