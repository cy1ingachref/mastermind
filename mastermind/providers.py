"""Provider adapters for MasterMind.

Each provider wraps a different AI model behind a common interface.
"""
from __future__ import annotations

import os
import json
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

    def __init__(self, api_key: str | None = None, model: str | None = None, **kwargs: Any):
        self.api_key = api_key or os.environ.get(self.env_var, "")
        self.model = model or self.model
        self.config = kwargs

    @property
    def is_available(self) -> bool:
        return not self.needs_key or bool(self.api_key)

    def complete(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        raise NotImplementedError


class ClaudeProvider(BaseProvider):
    """Anthropic Claude."""

    name = "claude"
    model = "claude-sonnet-4-20250514"
    env_var = "ANTHROPIC_API_KEY"
    url = "https://api.anthropic.com/v1/messages"

    def complete(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        messages = [{"role": "user", "content": prompt}]
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 4096,
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

    def complete(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
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

    def complete(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3},
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

    def complete(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
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

    def complete(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
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

    def complete(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
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
