"""Built-in tools for MasterMind agents."""
from __future__ import annotations

import os
import io
import sys
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.request import urlopen, Request
from urllib.parse import quote_plus

from . import Tool, ToolResult, ToolParameter, register_tool


class FileReadTool(Tool):
    """Read the contents of a file."""

    name = "file_read"
    description = "Read the contents of a file at the given path"
    parameters = [
        ToolParameter(
            name="path",
            type="string",
            description="Path to the file to read (relative or absolute)",
            required=True,
        ),
    ]

    def execute(self, **kwargs: Any) -> ToolResult:
        path = Path(kwargs["path"])
        if not path.exists():
            return ToolResult(success=False, output="", error=f"File not found: {path}")
        try:
            content = path.read_text(encoding="utf-8")
            return ToolResult(success=True, output=content)
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class FileWriteTool(Tool):
    """Write content to a file."""

    name = "file_write"
    description = "Write content to a file (creates parent directories if needed)"
    parameters = [
        ToolParameter(
            name="path",
            type="string",
            description="Path to the file to write",
            required=True,
        ),
        ToolParameter(
            name="content",
            type="string",
            description="Content to write to the file",
            required=True,
        ),
        ToolParameter(
            name="append",
            type="boolean",
            description="If true, append to the file instead of overwriting",
            required=False,
            default=False,
        ),
    ]

    def execute(self, **kwargs: Any) -> ToolResult:
        path = Path(kwargs["path"])
        content = kwargs["content"]
        append = kwargs.get("append", False)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if append:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(content)
            else:
                path.write_text(content, encoding="utf-8")
            return ToolResult(success=True, output=f"Wrote to {path}")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class FileListTool(Tool):
    """List files in a directory."""

    name = "file_list"
    description = "List files in a directory"
    parameters = [
        ToolParameter(
            name="path",
            type="string",
            description="Path to the directory (default: current directory)",
            required=False,
            default=".",
        ),
        ToolParameter(
            name="pattern",
            type="string",
            description="Glob pattern to filter files (e.g., '*.py')",
            required=False,
            default="*",
        ),
    ]

    def execute(self, **kwargs: Any) -> ToolResult:
        path = Path(kwargs.get("path", "."))
        pattern = kwargs.get("pattern", "*")
        if not path.exists():
            return ToolResult(success=False, output="", error=f"Directory not found: {path}")
        try:
            files = sorted(path.glob(pattern))
            output = "\n".join(str(f.relative_to(path)) for f in files)
            return ToolResult(success=True, output=output)
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class CodeExecuteTool(Tool):
    """Execute Python code in a sandbox."""

    name = "code_execute"
    description = "Execute Python code and return the output. Uses a temporary file."
    parameters = [
        ToolParameter(
            name="code",
            type="string",
            description="Python code to execute",
            required=True,
        ),
        ToolParameter(
            name="timeout",
            type="number",
            description="Timeout in seconds (default: 30, max: 60)",
            required=False,
            default=30,
        ),
    ]

    def execute(self, **kwargs: Any) -> ToolResult:
        code = kwargs["code"]
        timeout = min(kwargs.get("timeout", 30), 60)
        try:
            # Write code to temporary file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                temp_path = f.name

            # Execute with timeout
            start = time.time()
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            duration = time.time() - start

            output = result.stdout
            if result.stderr:
                output += f"\n[STDERR]\n{result.stderr}"

            os.unlink(temp_path)

            return ToolResult(
                success=result.returncode == 0,
                output=output or "(no output)",
                error="" if result.returncode == 0 else f"Exit code: {result.returncode}",
                duration=duration,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output="", error=f"Timeout after {timeout}s")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class WebFetchTool(Tool):
    """Fetch content from a URL."""

    name = "web_fetch"
    description = "Fetch content from a URL (HTML/text only, no JavaScript)"
    parameters = [
        ToolParameter(
            name="url",
            type="string",
            description="URL to fetch",
            required=True,
        ),
        ToolParameter(
            name="max_length",
            type="number",
            description="Maximum characters to return (default: 5000)",
            required=False,
            default=5000,
        ),
    ]

    def execute(self, **kwargs: Any) -> ToolResult:
        url = kwargs["url"]
        max_length = kwargs.get("max_length", 5000)
        try:
            req = Request(url, headers={"User-Agent": "MasterMind/0.1"})
            with urlopen(req, timeout=30) as resp:
                content = resp.read().decode("utf-8", errors="replace")[:max_length]
            return ToolResult(success=True, output=content)
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class WebSearchTool(Tool):
    """Search the web using DuckDuckGo."""

    name = "web_search"
    description = "Search the web for information"
    parameters = [
        ToolParameter(
            name="query",
            type="string",
            description="Search query",
            required=True,
        ),
    ]

    def execute(self, **kwargs: Any) -> ToolResult:
        query = kwargs["query"]
        try:
            # Use DuckDuckGo instant answer API
            url = f"https://api.duckduckgo.com/?q={quote_plus(query)}&format=json&no_html=1"
            req = Request(url, headers={"User-Agent": "MasterMind/0.1"})
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            # Extract abstract and related topics
            parts = []
            if data.get("AbstractText"):
                parts.append(f"Abstract: {data['AbstractText']}")
            if data.get("AbstractURL"):
                parts.append(f"Source: {data['AbstractURL']}")
            for topic in data.get("RelatedTopics", [])[:5]:
                if isinstance(topic, dict) and topic.get("Text"):
                    parts.append(f"- {topic['Text']}")

            output = "\n".join(parts) if parts else f"No results found for: {query}"
            return ToolResult(success=True, output=output)
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class ShellExecuteTool(Tool):
    """Execute a shell command."""

    name = "shell_execute"
    description = "Execute a shell command (use with caution)"
    parameters = [
        ToolParameter(
            name="command",
            type="string",
            description="Shell command to execute",
            required=True,
        ),
        ToolParameter(
            name="timeout",
            type="number",
            description="Timeout in seconds (default: 30, max: 120)",
            required=False,
            default=30,
        ),
    ]

    def execute(self, **kwargs: Any) -> ToolResult:
        command = kwargs["command"]
        timeout = min(kwargs.get("timeout", 30), 120)
        try:
            start = time.time()
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            duration = time.time() - start

            output = result.stdout
            if result.stderr:
                output += f"\n[STDERR]\n{result.stderr}"

            return ToolResult(
                success=result.returncode == 0,
                output=output or "(no output)",
                error="" if result.returncode == 0 else f"Exit code: {result.returncode}",
                duration=duration,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output="", error=f"Timeout after {timeout}s")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class TaskCompleteTool(Tool):
    """Signal that a task is complete."""

    name = "task_complete"
    description = "Signal that the current task is complete and provide the final result"
    parameters = [
        ToolParameter(
            name="result",
            type="string",
            description="The final result of the task",
            required=True,
        ),
        ToolParameter(
            name="summary",
            type="string",
            description="Brief summary of what was accomplished",
            required=False,
            default="",
        ),
    ]

    def execute(self, **kwargs: Any) -> ToolResult:
        result = kwargs["result"]
        summary = kwargs.get("summary", "")
        output = f"Task complete.\nResult: {result}"
        if summary:
            output = f"Summary: {summary}\n\n{output}"
        return ToolResult(success=True, output=output)


# Register all built-in tools
def register_defaults():
    """Register all built-in tools."""
    register_tool(FileReadTool())
    register_tool(FileWriteTool())
    register_tool(FileListTool())
    register_tool(CodeExecuteTool())
    register_tool(WebFetchTool())
    register_tool(WebSearchTool())
    register_tool(ShellExecuteTool())
    register_tool(TaskCompleteTool())


register_defaults()
