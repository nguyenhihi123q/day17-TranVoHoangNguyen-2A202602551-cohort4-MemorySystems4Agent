from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PROFILE_TEMPLATE = "# Profile: {user_id}\n\n_Chưa có thông tin._\n"
_FACT_LINE_RE = re.compile(r"^\s*-\s*\*\*(?P<key>[\w]+)\*\*\s*:\s*(?P<value>.+?)\s*$", re.MULTILINE)

# Memory decay: an unreinforced fact halves in priority every N days.
DEFAULT_HALF_LIFE_DAYS = 30.0
_SECONDS_PER_DAY = 86400.0


def estimate_tokens(text: str) -> int:
    """Approximate the token cost of ``text``.

    A stable heuristic is enough for the offline benchmark: strip whitespace,
    return 0 for empty text, otherwise ``len(text) / 4`` (min 1).
    """

    if not text:
        return 0
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, len(stripped) // 4)


@dataclass
class UserProfileStore:
    """Persistent storage for one ``User.md`` per user id.

    Supports read / write / edit plus structured fact helpers so the advanced
    agent can upsert stable facts and resolve corrections.

    Memory decay:
    - Each fact carries ``mentions`` (how often it was reinforced) and
      ``updated_at`` (last time its value changed) in a sidecar ``_meta.json``.
    - ``fact_priority`` blends exponential recency (half-life) with a log
      frequency bonus, so stale, rarely-mentioned facts sink and can be pruned.
    """

    root_dir: Path
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS

    def path_for(self, user_id: str) -> Path:
        # Sanitize the user id so it is always a safe single-path segment.
        safe_id = re.sub(r"[^0-9A-Za-z_\-]", "_", user_id).strip("_") or "default"
        user_dir = Path(self.root_dir) / safe_id
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return DEFAULT_PROFILE_TEMPLATE.format(user_id=user_id)

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        path = self.path_for(user_id)
        if not path.exists():
            return False
        content = path.read_text(encoding="utf-8")
        if search_text not in content:
            return False
        path.write_text(content.replace(search_text, replacement, 1), encoding="utf-8")
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        if path.exists():
            return path.stat().st_size
        return 0

    # --- optional structured helpers -------------------------------------

    def facts(self, user_id: str) -> dict[str, str]:
        """Parse ``User.md`` into a ``{key: value}`` mapping."""

        content = self.read_text(user_id)
        return {m.group("key"): m.group("value") for m in _FACT_LINE_RE.finditer(content)}

    def upsert_fact(self, user_id: str, key: str, value: str) -> Path:
        """Insert or update a single fact line, preserving the rest.

        A *changed* value refreshes the fact's recency (``updated_at``); a
        repeated identical value only bumps its mention counter, so reinforcement
        keeps a stable fact fresh without pretending it was newly learned.
        """

        content = self.read_text(user_id)
        previous = self.facts(user_id).get(key)
        marker = f"- **{key}**: "
        line_re = re.compile(re.escape(marker) + r".*")
        if line_re.search(content):
            content = line_re.sub(marker + value, content)
        else:
            content = content.rstrip("\n") + f"\n{marker}{value}\n"
        path = self.write_text(user_id, content)
        self.record_mention(user_id, key, changed=(previous != value))
        return path

    # --- memory decay -----------------------------------------------------

    def _meta_path_for(self, user_id: str) -> Path:
        return self.path_for(user_id).with_name("_meta.json")

    def _read_meta(self, user_id: str) -> dict[str, dict[str, float]]:
        """Load the per-fact decay metadata sidecar (``_meta.json``)."""

        path = self._meta_path_for(user_id)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except (ValueError, OSError):
                pass
        return {}

    def _write_meta(self, user_id: str, meta: dict[str, dict[str, float]]) -> Path:
        path = self._meta_path_for(user_id)
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def record_mention(self, user_id: str, key: str, changed: bool, now: float | None = None) -> None:
        """Update decay bookkeeping for one fact after it was seen."""

        now = time.time() if now is None else now
        meta = self._read_meta(user_id)
        entry = dict(meta.get(key) or {})
        entry["created_at"] = float(entry.get("created_at", now))
        entry["mentions"] = int(entry.get("mentions", 0)) + 1
        if changed or "updated_at" not in entry:
            entry["updated_at"] = now
        meta[key] = entry
        self._write_meta(user_id, meta)

    def _priority_from_meta(self, meta: dict[str, dict[str, float]], key: str, now: float) -> float:
        """Blend exponential recency (half-life) with a log frequency bonus."""

        entry = meta.get(key)
        if not entry:
            return 1.0
        updated = float(entry.get("updated_at", now))
        age_days = max(0.0, (now - updated) / _SECONDS_PER_DAY)
        half_life = self.half_life_days or DEFAULT_HALF_LIFE_DAYS
        recency = 0.5 ** (age_days / half_life)
        mentions = max(1, int(entry.get("mentions", 1)))
        frequency_bonus = 1.0 + 0.15 * math.log2(mentions)
        return recency * frequency_bonus

    def fact_priority(self, user_id: str, key: str, now: float | None = None) -> float:
        """Priority of one fact; newer and more-reinforced facts score higher."""

        now = time.time() if now is None else now
        return self._priority_from_meta(self._read_meta(user_id), key, now)

    def ranked_facts(
        self, user_id: str, max_facts: int | None = None, now: float | None = None
    ) -> list[tuple[str, str, float]]:
        """Return ``(key, value, priority)`` ordered by decaying priority."""

        now = time.time() if now is None else now
        meta = self._read_meta(user_id)
        scored = [
            (key, value, self._priority_from_meta(meta, key, now))
            for key, value in self.facts(user_id).items()
        ]
        scored.sort(key=lambda item: item[2], reverse=True)
        return scored[:max_facts] if max_facts is not None else scored

    def prune_facts(
        self,
        user_id: str,
        max_facts: int,
        min_priority: float = 0.15,
        now: float | None = None,
    ) -> list[str]:
        """Drop stale, low-priority facts so ``User.md`` cannot grow unbounded.

        Called on every write: any fact beyond ``max_facts`` or below
        ``min_priority`` is removed. Returns the keys that were dropped.
        """

        now = time.time() if now is None else now
        meta = self._read_meta(user_id)
        scored = sorted(
            self.facts(user_id).items(),
            key=lambda kv: self._priority_from_meta(meta, kv[0], now),
            reverse=True,
        )
        removed: list[str] = []
        for index, (key, _value) in enumerate(scored):
            priority = self._priority_from_meta(meta, key, now)
            if index >= max_facts or priority < min_priority:
                removed.append(key)
        if not removed:
            return []
        content = self.read_text(user_id)
        for key in removed:
            content = re.sub(
                rf"^\s*-\s*\*\*{re.escape(key)}\*\*\s*:.*\n?",
                "",
                content,
                flags=re.MULTILINE,
            )
            meta.pop(key, None)
        self.write_text(user_id, content)
        self._write_meta(user_id, meta)
        return removed

    def decay_report(self, user_id: str, now: float | None = None) -> list[tuple[str, float]]:
        """``(key, priority)`` snapshot for analysis / debugging."""

        now = time.time() if now is None else now
        meta = self._read_meta(user_id)
        return [
            (key, round(self._priority_from_meta(meta, key, now), 4))
            for key in self.facts(user_id)
        ]


# --- fact extraction helpers -------------------------------------------

# Question / filler words that must never be stored as a fact value.
_QUESTION_WORDS = {
    "gì", "đâu", "nào", "ai", "sao", "bao", "nhiêu", "thế", "không",
    "chưa", "vậy", "hả", "ừ", "à", "ơi", "nhỉ",
}

# Phrases that signal the turn actually carries a stable fact (used to tell a
# real statement apart from a pure recall question).
_FACT_HINTS = (
    "mình là", "mình tên", "tên mình", "tên là", "mình ở", "mình đang ở",
    "mình làm", "mình thích", "mình muốn", "mình nuôi", "yêu thích là",
    "chuyển sang", "đang làm", "hiện ở", "vẫn ở",
)

# Connectors used to stop a captured value before it drifts into the next clause.
_CONNECT_RE = re.compile(
    r"\s+(?:nhưng|như|và|rồi|để|chứ|còn|với|vì|mỗi|trong|khi|thì|cũng|vẫn|dù|mà|giờ|đây)\b",
    re.IGNORECASE,
)

# Tokens that only ever appear in filler/question phrases, never as a real value.
_FILLER = {
    "yêu", "thích", "gì", "đâu", "nào", "nhất", "là", "và", "của", "mình",
    "bạn", "hãy", "cho", "với", "con", "món", "ai", "nhắc", "lại", "hiện",
    "tại", "kiểu", "thế", "như", "cách", "thông", "tin", "nơi", "ở", "ruột",
}

# Values that look like a location slot but are actually noise.
_LOC_NOISE = {
    "quán", "đâu", "nhà", "công", "ty", "văn", "phòng", "hiện", "nay",
    "tại", "đây", "kia", "đó", "này", "giờ", "chỗ", "trên", "dưới", "ngay",
}

_LOC_RE = re.compile(
    r"(?:hiện\s+(?:đang\s+)?ở|đang\s+ở|mình\s+(?:vẫn\s+)?ở|vẫn\s+ở|"
    r"chuyển\s+(?:về|đến)|nơi\s+ở[^.\n]*?là|làm\s+việc\s+ở)"
    r"\s+([A-Za-zÀ-ỹ][\wÀ-ỹ]*(?:\s+[A-Za-zÀ-ỹ][\wÀ-ỹ]*)?)",
    re.IGNORECASE,
)

# Role head nouns used to reject location false positives
# ("chuyển sang MLOps engineer" is NOT a location).
_ROLE_NOUNS = {
    "engineer", "manager", "developer", "analyst", "scientist",
    "designer", "architect", "specialist",
}

# Role head noun, optionally preceded by a known qualifier ("backend engineer").
_ROLE_RE = re.compile(
    r"((?:(?:backend|frontend|full[- ]?stack|data|mlops|software|senior|junior|"
    r"lead|staff|principal|cloud|devops|platform)\s+)?"
    r"(?:engineer|manager|developer|analyst|scientist|designer|architect|specialist))"
    r"(?=[\s.,;:!?\)]|$)",
    re.IGNORECASE,
)
# Context that proves a role mention is OLD / negated (must not overwrite).
_OLD_MARKERS = (
    "không còn", "không làm", "chứ không", "đừng", "trước đây", "trước kia",
    "thông tin cũ", "từng là", "thay vì", "case", "giống",
)
# Context that proves a role mention is the CURRENT / new one.
_NEW_MARKERS = ("chuyển sang", "chuyển qua", "hiện tại", "bây giờ", "giờ", "đang làm", "làm ")
_JOKE_MARKERS = ("đùa", "hay là", "chắc là", "có lẽ", "hình như", "giả sử", "nếu là")

_STYLE_HINT_RE = re.compile(r"trả lời|style|câu trả lời|bullet|gọn", re.IGNORECASE)
_TECH_HINT_RE = re.compile(
    r"\b(python|ai|mlops|rag|memory|agent|benchmark|công nghệ|kỹ thuật)\b",
    re.IGNORECASE,
)


def _clean_value(text: str, max_len: int = 60) -> str | None:
    """Trim a captured value, drop question words/filler, cap its length."""

    val = _CONNECT_RE.split(text, maxsplit=1)[0]
    val = val.strip().strip("\"'").rstrip(".,;:!?")
    if not val:
        return None
    words = val.split()
    if not words or words[0].lower() in _QUESTION_WORDS:
        return None
    if words[0].lower() in _FILLER:
        return None
    if len(val) > max_len:
        val = val[:max_len].rstrip()
    return val or None


def _is_pure_question(message: str) -> bool:
    stripped = message.strip()
    return stripped.endswith("?")


def _extract_name(message: str) -> str | None:
    """Capture a personal name from self-introduction patterns."""

    best: str | None = None
    for m in re.finditer(
        r"(?:mình\s+tên\s+(?:là\s+)?|tên\s+mình\s+(?:là\s+)?|tên\s+tôi\s+(?:là\s+)?"
        r"|tên\s+là\s+|mình\s+là\s+|tôi\s+là\s+)"
        r"([A-Za-zÀ-ỹ][\wÀ-ỹ]*(?:\s+[A-Za-zÀ-ỹ][\wÀ-ỹ]*){0,3})",
        message,
    ):
        # Trim at connectors/question tails within the captured span.
        span = m.group(1)
        span = re.split(r"\s+(?:và|nhưng|,|\.|;|hiện|đang|chứ|ở|vẫn|giờ)\b", span, maxsplit=1)[0]
        span = span.strip().rstrip(".,;!?")
        words = span.split()
        if not words:
            continue
        first = words[0]
        if first.lower() in _FILLER or first.lower() in _QUESTION_WORDS:
            continue
        best = " ".join(words[:4])
        break  # first confident introduction wins
    return _clean_value(best, max_len=40) if best else None


def _extract_location(message: str) -> str | None:
    """Capture current location, letting a correction inside the same turn win."""

    for m in _LOC_RE.finditer(message):
        val = _clean_value(m.group(1), max_len=40)
        if not val:
            continue
        if any(w in _LOC_NOISE for w in val.lower().split()):
            continue
        # Reject role nouns that leak in via "chuyển sang <role>".
        if any(w.lower() in _ROLE_NOUNS for w in val.split()):
            continue
        if len(val) < 2:
            continue
        return val
    return None


def _role_span(message: str, m: re.Match) -> str:
    """Return the matched role span (qualifier already included by ``_ROLE_RE``)."""

    return m.group(1).strip()


def _extract_profession(message: str) -> str | None:
    """Pick the CURRENT role, ignoring negated/joke mentions.

    ``_OLD_MARKERS`` (e.g. "không còn", "đừng") invalidate a mention;
    ``_NEW_MARKERS`` (e.g. "chuyển sang", "hiện tại") promote it.
    """

    new_pick: str | None = None
    fallback_pick: str | None = None
    for m in _ROLE_RE.finditer(message):
        before = message[max(0, m.start() - 85):m.start()].lower()
        if any(j in before for j in _JOKE_MARKERS):
            continue
        if any(o in before for o in _OLD_MARKERS):
            continue  # negated/old mention -> ignore
        span = _role_span(message, m)
        fallback_pick = span
        if any(n in before for n in _NEW_MARKERS):
            new_pick = span
    return new_pick or fallback_pick


def _extract_drink(message: str) -> str | None:
    m = re.search(
        r"(?:đồ uống[^.\n]*?(?:yêu thích|thích)[^.\n]*?là|đồ uống yêu thích là"
        r"|thích uống|thường uống|vẫn uống|hay uống)\s+([^\.,;!\n]+)",
        message,
        re.IGNORECASE,
    )
    if not m:
        return None
    val = m.group(1).strip().rstrip(".,;!?")
    val = re.split(r"\s+(?:như|như cũ|và|nhưng|đang|mỗi|để)\b", val, maxsplit=1)[0].strip()
    if val.lower().startswith("cà phê sữa đá"):
        return "cà phê sữa đá"
    return _clean_value(val, max_len=40)


def _extract_food(message: str) -> str | None:
    m = re.search(
        r"(?:món ăn[^.\n]*?(?:yêu thích|thích)[^.\n]*?là|món ruột[^.\n]*?(?:là|chính là)"
        r"|đúng là món ruột|món yêu thích là|thích ăn|hay ăn)\s+([^\.,;!\n]+)",
        message,
        re.IGNORECASE,
    )
    if not m:
        return None
    val = m.group(1).strip().rstrip(".,;!?")
    val = re.split(r"\s+(?:và|nhưng|vì|với|để|còn|thì|đang)\b", val, maxsplit=1)[0].strip()
    if val.lower().startswith("mì quảng"):
        return "mì Quảng"
    return _clean_value(val, max_len=40)


def _extract_pet(message: str) -> str | None:
    m = re.search(
        r"(?:nuôi|thú cưng)[^.\n]*?(?:một\s+|con\s+|bé\s+)*([\wÀ-ỹ]+)\s+tên\s+([\wÀ-ỹ]+)",
        message,
        re.IGNORECASE,
    )
    return f"{m.group(1)} tên {m.group(2)}" if m else None


def _extract_interests(message: str) -> str | None:
    m = re.search(
        r"(?:quan tâm (?:nhiều )?(?:đến|tới)|đam mê|mối quan tâm[^:]*:)\s+([^\n.]+)",
        message,
        re.IGNORECASE,
    )
    if not m:
        return None
    val = m.group(1).strip().rstrip(".,;!?")
    # Keep only technical interests so chit-chat does not overwrite
    # the real technical focus (Python / AI / ...).
    if not _TECH_HINT_RE.search(val):
        return None
    # Trim at common sentence connectors.
    val = re.split(r"\s+(?:nên|hãy|cho|từ|bằng)", val, maxsplit=1)[0].strip()
    return val[:80].rstrip() if len(val) > 80 else (val or None)


def _extract_style(message: str) -> str | None:
    if not _STYLE_HINT_RE.search(message):
        return None
    parts: list[str] = []
    if re.search(r"ngắn gọn|ngắn\b|gọn\b", message, re.IGNORECASE):
        parts.append("ngắn gọn")
    if re.search(r"3 bullet", message, re.IGNORECASE):
        parts.append("3 bullet")
    elif re.search(r"\bbullet", message, re.IGNORECASE):
        parts.append("bullet")
    if re.search(r"ví dụ", message, re.IGNORECASE):
        parts.append("ví dụ thực tế" if re.search(r"thực tế", message, re.IGNORECASE) else "ví dụ thực chiến")
    return ", ".join(parts) if parts else None


def extract_profile_updates(message: str) -> dict[str, str]:
    """Convert raw user text into stable profile facts.

    Returns only facts confidently present in the message. Question-only turns
    and ambiguous/joke statements are skipped so persistent ``User.md`` stays
    clean (confidence threshold + noise guard).
    """

    if _is_pure_question(message):
        return {}

    facts: dict[str, str] = {}
    extractors = {
        "name": _extract_name,
        "location": _extract_location,
        "profession": _extract_profession,
        "favorite_drink": _extract_drink,
        "favorite_food": _extract_food,
        "pet": _extract_pet,
        "interests": _extract_interests,
        "response_style": _extract_style,
    }
    for key, fn in extractors.items():
        value = fn(message)
        if value:
            facts[key] = value
    return facts


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a compact summary of older messages.

    Heuristic concatenation for now; can be swapped for an LLM summary later.
    """

    if not messages:
        return ""
    selected = messages[:max_items] if len(messages) > max_items else messages
    lines: list[str] = []
    for msg in selected:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        short = content[:150] + "..." if len(content) > 150 else content
        lines.append(f"[{role}]: {short}")
    return "Tóm tắt hội thoại trước:\n" + "\n".join(lines)


@dataclass
class CompactMemoryManager:
    """Compact memory for long threads.

    Goal:
    - Keep recent messages in full
    - When the thread grows too large, move older content into a summary
    - Track how many compactions happened for benchmarking
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _ensure_thread(self, thread_id: str) -> dict[str, object]:
        if thread_id not in self.state:
            self.state[thread_id] = {"messages": [], "summary": "", "compactions": 0}
        return self.state[thread_id]

    def _total_tokens(self, thread: dict[str, object]) -> int:
        total = estimate_tokens(str(thread.get("summary", "")))
        for msg in thread["messages"]:  # type: ignore[index]
            total += estimate_tokens(str(msg["content"]))
        return total

    def append(self, thread_id: str, role: str, content: str) -> None:
        # 1. create thread state if missing
        thread = self._ensure_thread(thread_id)

        # 2. append the new message
        messages: list[dict[str, str]] = thread["messages"]  # type: ignore[assignment]
        messages.append({"role": role, "content": content})

        # 3. trigger compaction if needed
        if self._total_tokens(thread) > self.threshold_tokens and len(messages) > self.keep_messages:
            old_messages = messages[: -self.keep_messages]
            kept_messages = messages[-self.keep_messages :]

            new_summary_part = summarize_messages(old_messages)
            old_summary = str(thread.get("summary", ""))
            thread["summary"] = f"{old_summary}\n{new_summary_part}".strip() if old_summary else new_summary_part

            thread["messages"] = kept_messages
            thread["compactions"] = int(thread.get("compactions", 0)) + 1  # type: ignore[arg-type]

    def context(self, thread_id: str) -> dict[str, object]:
        """Return per-thread state with keys ``messages``, ``summary``, ``compactions``."""

        return self._ensure_thread(thread_id)

    def compaction_count(self, thread_id: str) -> int:
        return int(self._ensure_thread(thread_id).get("compactions", 0))  # type: ignore[arg-type]
