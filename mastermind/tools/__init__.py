"""Tool registry and base class.

Agents use tools to interact with the world:
- File operations (read/write/edit)
- Code execution
- Web search/fetch
- Custom user-defined tools
"""
from __future__ import annotations

import json
import time
from typing import Any, Callable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ToolParameter:
    """A parameter for a tool."""
    name: str
    type: str = "string"
    description: str = ""
    required: bool = True
    default: Any = None


@dataclass
class ToolResult:
    """Result of a tool execution."""
    success: bool
    output: str
    error: str = ""
    duration: float = 0.0


class Tool:
    """Base class for tools that agents can use."""

    name: str = "base"
    description: str = ""
    parameters: list[ToolParameter] = []

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with given parameters."""
        raise NotImplementedError

    def to_schema(self) -> dict[str, Any]:
        """Return tool schema."""
        properties = {}
        required = []
        for param in self.parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description,
            }
            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


class ToolRegistry:
    """Registry of available tools."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        """List all registered tools."""
        return list(self._tools.values())

    def get_schemas(self) -> list[dict[str, Any]]:
        """Get schemas for all tools."""
        return [tool.to_schema() for tool in self._tools.values()]

    def execute(self, name: str, parameters: dict[str, Any]) -> ToolResult:
        """Execute a tool by name."""
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(
                success=False,
                output="",
                error=f"Tool '{name}' not found",
            )
        return tool.execute(**parameters)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


# Global registry
_registry = ToolRegistry()


def get_registry() -> ToolRegistry:
    """Get the global tool registry."""
    return _registry


def register_tool(tool: Tool) -> None:
    """Register a tool in the global registry."""
    _registry.register(tool)


# Agent loop
from .agent_loop import AgentLoop, ToolCall

# Register built-in tools
from . import builtins

__all__ = [
    "Tool",
    "ToolResult",
    "ToolParameter",
    "ToolRegistry",
    "AgentLoop",
    "ToolCall",
    "register_tool",
    "get_registry",
]
