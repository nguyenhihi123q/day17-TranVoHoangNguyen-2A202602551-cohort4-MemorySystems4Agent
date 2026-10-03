from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig


@dataclass
class LabConfig:
    """Shared configuration for the lab.

    - Keeps paths for the repo root, dataset directory, and state directory.
    - Holds compact-memory settings such as threshold and messages to keep.
    - Holds provider settings for the main model and the judge model.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig
    # Memory-decay settings for the persistent profile.
    max_profile_facts: int = 12
    memory_half_life_days: float = 30.0


def _first_env(*names: str) -> str | None:
    """Return the first non-empty environment variable from ``names``."""

    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load environment variables and return a populated ``LabConfig``.

    Steps:
    1. Resolve the repo root (or default to the current file's parent.parent).
    2. Optionally load values from `.env` (via python-dotenv, if installed).
    3. Create `state/` if it does not exist.
    4. Build the main model + judge model provider configs.
    """

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    # Optional .env loading. python-dotenv may not be installed; ignore if so.
    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env", override=False)
    except ImportError:
        pass

    data_dir = root / "data"
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    provider = os.getenv("LLM_PROVIDER", "openai")
    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # Allow the judge to run on a different (usually cheaper) model.
    judge_model_name = os.getenv("JUDGE_MODEL", model_name)
    judge_provider = os.getenv("JUDGE_PROVIDER", provider)

    api_key = _first_env(
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENROUTER_API_KEY",
        "CUSTOM_API_KEY",
    )
    base_url = _first_env("CUSTOM_BASE_URL", "OLLAMA_BASE_URL")

    model_cfg = ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=0.3,
        api_key=api_key,
        base_url=base_url,
    )
    judge_cfg = ProviderConfig(
        provider=judge_provider,
        model_name=judge_model_name,
        temperature=0.0,
        api_key=api_key,
        base_url=base_url,
    )

    # Compact-memory defaults: compact once a thread carries more than ~1500
    # tokens of context, then keep the 6 most recent messages in full.
    compact_threshold_tokens = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "1500"))
    compact_keep_messages = int(os.getenv("COMPACT_KEEP_MESSAGES", "6"))

    # Memory decay: cap how many facts reach the prompt and how fast an unreinforced
    # fact stops being "fresh" (exponential half-life).
    max_profile_facts = int(os.getenv("MAX_PROFILE_FACTS", "12"))
    memory_half_life_days = float(os.getenv("MEMORY_HALF_LIFE_DAYS", "30"))

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold_tokens,
        compact_keep_messages=compact_keep_messages,
        model=model_cfg,
        judge_model=judge_cfg,
        max_profile_facts=max_profile_facts,
        memory_half_life_days=memory_half_life_days,
    )
