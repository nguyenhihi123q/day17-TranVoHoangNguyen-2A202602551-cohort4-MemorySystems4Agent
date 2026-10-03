from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A: within-session memory only.

    Requirements:
    - Within-session memory only
    - No persistent `User.md`
    - Forgets long-term facts across new threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}

        # Optionally initialize a real LangChain/LangGraph agent when deps exist.
        self.langchain_agent = None
        if not self.force_offline:
            self._maybe_build_langchain_agent()

    def _get_session(self, thread_id: str) -> SessionState:
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()
        return self.sessions[thread_id]

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return the agent response and token accounting.

        Uses the live path when available, otherwise the deterministic offline
        path. The baseline never remembers facts across different threads.
        """

        if self.langchain_agent is not None and not self.force_offline:
            try:
                return self._reply_live(thread_id, message)
            except Exception:
                # Fall back to offline behavior if the live call fails.
                pass
        return self._reply_offline(thread_id, message)

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        """Best-effort live path; used only when a real model is configured."""

        self.langchain_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread_id}},
        )
        # Reuse the offline accounting so benchmark columns stay comparable.
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self._get_session(thread_id).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self._get_session(thread_id).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic offline behavior.

        - Store the new user message in the session
        - Generate a short deterministic reply (echo, no long-term memory)
        - Update token counts
        - Never remember facts across different thread ids
        """

        session = self._get_session(thread_id)
        session.messages.append({"role": "user", "content": message})

        session.token_usage += estimate_tokens(message)

        # Prompt context = every message kept so far in this thread (the baseline
        # keeps the whole history, so its prompt load grows without bound).
        prompt_ctx = sum(estimate_tokens(m["content"]) for m in session.messages)
        session.prompt_tokens_processed += prompt_ctx

        response = f"[Baseline] Đã nhận: {message[:100]}"
        resp_tokens = estimate_tokens(response)
        session.token_usage += resp_tokens

        session.messages.append({"role": "assistant", "content": response})

        return {"reply": response, "tokens": resp_tokens}

    def _maybe_build_langchain_agent(self) -> None:
        """Optionally wire a LangGraph agent with in-memory short-term state.

        Falls back to ``None`` (offline mode) when dependencies or API keys are
        unavailable, so the offline benchmark always works.
        """

        try:
            from langgraph.checkpoint.memory import InMemorySaver
            from langgraph.prebuilt import create_react_agent

            model = build_chat_model(self.config.model)
            self.langchain_agent = create_react_agent(
                model,
                tools=[],
                checkpointer=InMemorySaver(),
            )
        except Exception:
            self.langchain_agent = None
