"""Provider adapters for MasterMind.

Each provider wraps a different AI model behind a common interface.
All providers support cost tracking and retry with exponential backoff.
"""
from __future__ import annotations

import os
import time
import random
from typing import Any

import requests
from rich.console import Console

console = Console()


class BaseProvider:
    """Base class for AI providers."""

    name: str = "base"
    model: str = ""
    needs_key: bool = True
    env_var: str = ""

    # Cost per 1M tokens (input, output)
    cost_per_1m: tuple[float, float] = (0.0, 0.0)

    def __init__(self, api_key: str | None = None, model: str | None = None, **kwargs: Any):
        self.api_key = api_key or os.environ.get(self.env_var, "")
        self.model = model or self.model
        self.config = kwargs

    @property
    def is_available(self) -> bool:
        return not self.needs_key or bool(self.api_key)

    def complete(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        raise NotImplementedError

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost in USD."""
        input_cost = (prompt_tokens / 1_000_000) * self.cost_per_1m[0]
        output_cost = (completion_tokens / 1_000_000) * self.cost_per_1m[1]
        return input_cost + output_cost


class ClaudeProvider(BaseProvider):
    """Anthropic Claude."""

    name = "claude"
    model = "claude-sonnet-4-20250514"
    env_var = "ANTHROPIC_API_KEY"
    url = "https://api.anthropic.com/v1/messages"
    cost_per_1m = (3.0, 15.0)

    def complete(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        messages = [{"role": "user", "content": prompt}]
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
        }
        if system:
            payload["system"] = system

        resp = requests.post(
            self.url,
            headers={
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            },
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"].strip()


class GPTProvider(BaseProvider):
    """OpenAI GPT."""

    name = "gpt"
    model = "gpt-4o"
    env_var = "OPENAI_API_KEY"
    url = "https://api.openai.com/v1/chat/completions"
    cost_per_1m = (2.5, 10.0)

    def complete(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.3),
            "max_tokens": kwargs.get("max_tokens", 4096),
        }

        resp = requests.post(
            self.url,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


class GeminiProvider(BaseProvider):
    """Google Gemini."""

    name = "gemini"
    model = "gemini-2.5-flash"
    env_var = "GOOGLE_API_KEY"
    url = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    cost_per_1m = (0.15, 0.6)

    def complete(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": kwargs.get("temperature", 0.3)},
        }
        if system:
            payload["system_instruction"] = {"parts": [{"text": system}]}

        api_url = self.url.format(model=self.model)
        resp = requests.post(
            api_url,
            headers={"Content-Type": "application/json"},
            json=payload,
            params={"key": self.api_key},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()


class KimiProvider(BaseProvider):
    """Moonshot Kimi."""

    name = "kimi"
    model = "moonshot-v1-8k"
    env_var = "MOONSHOT_API_KEY"
    url = "https://api.moonshot.cn/v1/chat/completions"
    cost_per_1m = (1.0, 2.0)

    def complete(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.3),
        }

        resp = requests.post(
            self.url,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


class GrokProvider(BaseProvider):
    """xAI Grok."""

    name = "grok"
    model = "grok-2-latest"
    env_var = "XAI_API_KEY"
    url = "https://api.x.ai/v1/chat/completions"
    cost_per_1m = (2.0, 10.0)

    def complete(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.3),
        }

        resp = requests.post(
            self.url,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


class MistralProvider(BaseProvider):
    """Mistral AI."""

    name = "mistral"
    model = "mistral-large-latest"
    env_var = "MISTRAL_API_KEY"
    url = "https://api.mistral.ai/v1/chat/completions"
    cost_per_1m = (2.0, 6.0)

    def complete(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.3),
        }

        resp = requests.post(
            self.url,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


# ─── Provider registry ───────────────────────────────────────────────────────

PROVIDERS = {
    "claude": ClaudeProvider,
    "gpt": GPTProvider,
    "gemini": GeminiProvider,
    "kimi": KimiProvider,
    "grok": GrokProvider,
    "mistral": MistralProvider,
}


def get_provider(name: str, **kwargs: Any) -> BaseProvider:
    """Get a provider by name."""
    name = name.lower()
    if name not in PROVIDERS:
        raise ValueError(f"Unknown provider: {name}. Available: {', '.join(PROVIDERS.keys())}")
    return PROVIDERS[name](**kwargs)


def list_providers() -> list[str]:
    """List available provider names."""
    return list(PROVIDERS.keys())


def list_available_providers() -> list[str]:
    """List providers that have API keys set."""
    available = []
    for name, provider_class in PROVIDERS.items():
        instance = provider_class()
        if instance.is_available:
            available.append(name)
    return available


def retry_with_backoff(
    fn,
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
) -> Any:
    """Execute function with exponential backoff retry."""
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                # Add jitter
                delay = delay * (0.5 + random.random() * 0.5)
                console.print(f"[yellow]Retry {attempt + 1}/{max_retries} after {delay:.1f}s: {e}[/yellow]")
                time.sleep(delay)
    raise last_exception
