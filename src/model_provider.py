from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Provider configuration shared by the agents.

    Supported providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


# Aliases / common typos mapped to the canonical provider name.
_PROVIDER_ALIASES: dict[str, str] = {
    "anthorpic": "anthropic",
    "anthropic_claude": "anthropic",
    "claude": "anthropic",
    "open_ai": "openai",
    "open-ai": "openai",
    "google": "gemini",
    "google-genai": "gemini",
    "google_genai": "gemini",
    "gemini-ai": "gemini",
    "open-router": "openrouter",
    "open_router": "openrouter",
    "local": "ollama",
}


def normalize_provider(value: str) -> str:
    """Map provider aliases such as ``anthorpic`` -> ``anthropic``.

    Lowercases and strips whitespace so callers can pass values like
    ``" A nthorpic "`` without crashing.
    """

    if not value:
        return ""
    v = value.strip().lower()
    return _PROVIDER_ALIASES.get(v, v)


def build_chat_model(config: ProviderConfig):
    """Instantiate the real chat model for the selected provider.

    The live path is optional for this lab (benchmark/tests run offline), so
    provider SDKs are imported lazily to avoid hard dependencies at import time.
    """

    provider = normalize_provider(config.provider)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
        )
    elif provider == "custom":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
            base_url=config.base_url,
        )
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            google_api_key=config.api_key,
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=config.model_name,
            temperature=config.temperature,
            anthropic_api_key=config.api_key,
        )
    elif provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=config.model_name,
            temperature=config.temperature,
            base_url=config.base_url,
        )
    elif provider == "openrouter":
        from langchain_openrouter import ChatOpenRouter

        return ChatOpenRouter(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")
