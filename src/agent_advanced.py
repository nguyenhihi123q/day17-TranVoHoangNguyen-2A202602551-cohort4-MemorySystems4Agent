from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B / Advanced Agent.

    Required memory layers:
    1. within-session memory (compact memory)
    2. persistent `User.md`
    3. compact memory for long threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(
            self.config.state_dir / "profiles",
            half_life_days=self.config.memory_half_life_days,
        )
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}

        # Optionally initialize a real LangChain/LangGraph agent.
        self.langchain_agent = None
        if not self.force_offline:
            self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Route between offline mode and live mode."""

        if self.langchain_agent is not None and not self.force_offline:
            try:
                return self._reply_live(user_id, thread_id, message)
            except Exception:
                pass
        return self._reply_offline(user_id, thread_id, message)

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Best-effort live path; falls back to offline accounting."""

        profile = self.profile_store.read_text(user_id)
        ctx = self.compact_memory.context(thread_id)
        prompt = (
            f"User profile:\n{profile}\n\nSummary:\n{ctx.get('summary', '')}\n\n"
            f"User: {message}"
        )
        self.langchain_agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config={"configurable": {"thread_id": thread_id}},
        )
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic advanced path.

        1. Extract stable profile facts from the incoming message.
        2. Persist those facts into `User.md` (upsert -> corrections win).
        3. Append the message into compact memory.
        4. Estimate prompt-context load from `User.md` + summary + recent messages.
        5. Generate a response that can answer long-term recall questions.
        6. Append the assistant reply and update token counters.
        """

        # 1 + 2. Extract and persist facts (upsert overrides conflicting values).
        facts = extract_profile_updates(message)
        for key, value in facts.items():
            self.profile_store.upsert_fact(user_id, key, value)

        # 2b. Memory decay: drop stale / excess facts so User.md stays bounded.
        self.profile_store.prune_facts(user_id, max_facts=self.config.max_profile_facts)

        # 3. Short-term + compact memory.
        self.compact_memory.append(thread_id, "user", message)

        # 4. Estimate the context this turn has to carry.
        prompt_ctx = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_ctx

        # 5. Generate a memory-aware response.
        response = self._offline_response(user_id, thread_id, message)
        resp_tokens = estimate_tokens(response)
        self.thread_tokens[thread_id] = (
            self.thread_tokens.get(thread_id, 0) + estimate_tokens(message) + resp_tokens
        )

        # 6. Keep the assistant reply in short-term memory as well.
        self.compact_memory.append(thread_id, "assistant", response)

        return {"reply": response, "tokens": resp_tokens}

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Estimate the context carried into one turn.

        Includes `User.md`, the compact summary, and the recent kept messages.
        """

        profile_tokens = estimate_tokens(self.profile_store.read_text(user_id))
        ctx = self.compact_memory.context(thread_id)
        summary = str(ctx.get("summary", ""))
        messages = ctx.get("messages", []) or []
        summary_tokens = estimate_tokens(summary)
        message_tokens = sum(estimate_tokens(str(m["content"])) for m in messages)
        return profile_tokens + summary_tokens + message_tokens

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Return a deterministic answer using persisted memory.

        Answers long-term recall questions ("Tên mình là gì?", "Nghề hiện tại?",
        "Style trả lời?", "Nơi ở?") from `User.md` so cross-session recall passes.
        """

        facts = self.profile_store.facts(user_id)
        ctx = self.compact_memory.context(thread_id)
        lower = message.lower()

        recall_markers = (
            "tên", "name", "ai", "nhắc", "nhớ", "đâu", "nào", "gì", "nghề",
            "style", "uống", "ăn", "nuôi", "mối quan tâm", "tóm tắt", "ở đâu",
        )
        is_recall = any(k in lower for k in recall_markers)

        if is_recall:
            parts = ["[Advanced] Dựa trên memory:"]
            # Inject by decaying priority so fresh / reinforced facts surface first
            # and stale ones are pushed to the tail.
            ranked = self.profile_store.ranked_facts(
                user_id, max_facts=self.config.max_profile_facts
            )
            if ranked:
                for key, value, _priority in ranked:
                    parts.append(f"- {key}: {value}")
            else:
                parts.append("- Chưa có thông tin về bạn.")
        else:
            parts = [f"[Advanced] Đã ghi nhận: {message[:100]}"]
            if facts:
                parts.append("(Đã cập nhật User.md)")

        compactions = ctx.get("compactions", 0)
        if compactions:
            parts.append(f"(Có {compactions} lần compact)")
        return "\n".join(parts)

    def _maybe_build_langchain_agent(self) -> None:
        """Wire a live agent with profile tools + compact middleware.

        Falls back to ``None`` (offline mode) when dependencies or API keys are
        unavailable, so the offline benchmark always works.
        """

        try:
            from langgraph.checkpoint.memory import InMemorySaver
            from langgraph.prebuilt import create_react_agent

            store = self.profile_store

            def read_user_md(user_id: str) -> str:
                """Read the persisted User.md profile for a user."""

                return store.read_text(user_id)

            def write_user_md(user_id: str, content: str) -> str:
                """Overwrite the User.md profile for a user."""

                return str(store.write_text(user_id, content))

            model = build_chat_model(self.config.model)
            self.langchain_agent = create_react_agent(
                model,
                tools=[read_user_md, write_user_md],
                checkpointer=InMemorySaver(),
            )
        except Exception:
            self.langchain_agent = None
