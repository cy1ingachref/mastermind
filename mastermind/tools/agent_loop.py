"""Agent loop with tool execution and multi-step reasoning."""
from __future__ import annotations

import json
import time
from typing import Any, Protocol

from rich.console import Console

from . import ToolRegistry, ToolResult

console = Console()


class ProviderProtocol(Protocol):
    """Protocol for providers that can complete prompts."""
    def complete(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str: ...


class ToolCall:
    """A tool call request from the agent."""

    def __init__(self, name: str, arguments: dict[str, Any]):
        self.name = name
        self.arguments = arguments


class AgentLoop:
    """Agent that can use tools to accomplish tasks."""

    def __init__(
        self,
        provider: Any,  # Any object with a complete method
        registry: ToolRegistry,
        system_prompt: str = "",
        max_steps: int = 10,
    ):
        self.provider = provider
        self.registry = registry
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.history: list[dict[str, str]] = []

    def run(self, task: str) -> tuple[str, list[dict[str, Any]]]:
        """Run the agent loop to complete a task."""
        trace = []
        self.history = [{"role": "user", "content": task}]

        for step in range(self.max_steps):
            response = self._get_action()
            tool_call = self._parse_tool_call(response)

            if tool_call:
                result = self.registry.execute(tool_call.name, tool_call.arguments)
                trace.append({
                    "step": step,
                    "type": "tool_call",
                    "tool": tool_call.name,
                    "arguments": tool_call.arguments,
                    "result": result.output,
                    "success": result.success,
                    "duration": result.duration,
                })
                self.history.append({
                    "role": "user",
                    "content": f"Tool '{tool_call.name}' result:\n{result.output}",
                })
                if tool_call.name == "task_complete":
                    return result.output, trace
            else:
                trace.append({
                    "step": step,
                    "type": "response",
                    "content": response,
                })
                return response, trace

        return "Max steps reached without completion.", trace

    def _get_action(self) -> str:
        """Get the next action from the agent."""
        tool_descriptions = self._format_tools()
        full_system = f"{self.system_prompt}\n\n{tool_descriptions}" if self.system_prompt else tool_descriptions
        return self.provider.complete(
            prompt=self._format_messages(),
            system=full_system,
        )

    def _format_tools(self) -> str:
        """Format tool descriptions for the agent."""
        schemas = self.registry.get_schemas()
        if not schemas:
            return ""
        parts = ["You have access to the following tools:"]
        for schema in schemas:
            params = schema.get("parameters", {}).get("properties", {})
            param_list = ", ".join(f"{k}({v.get('type', 'any')})" for k, v in params.items())
            parts.append(f"- {schema['name']}({param_list}): {schema['description']}")
        parts.append('\nTo use a tool, respond with JSON: {"tool": "name", "arguments": {"param": "value"}}')
        return "\n".join(parts)

    def _format_messages(self) -> str:
        """Format message history for the provider."""
        parts = []
        for msg in self.history:
            role = msg["role"].upper()
            parts.append(f"{role}: {msg['content']}")
        return "\n\n".join(parts)

    def _parse_tool_call(self, response: str) -> ToolCall | None:
        """Parse a tool call from the response."""
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
                if "tool" in data and "arguments" in data:
                    return ToolCall(data["tool"], data["arguments"])
        except (json.JSONDecodeError, Exception):
            pass
        return None
