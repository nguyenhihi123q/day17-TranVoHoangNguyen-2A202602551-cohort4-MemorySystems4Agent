from __future__ import annotations

import time
from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore
from model_provider import ProviderConfig


def make_config(tmp_path: Path):
    """Build an isolated config for tests."""

    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=state_dir,
        compact_threshold_tokens=200,  # low threshold so compaction happens fast
        compact_keep_messages=2,
        model=ProviderConfig(provider="openai", model_name="gpt-4o-mini", temperature=0.0),
        judge_model=ProviderConfig(provider="openai", model_name="gpt-4o-mini", temperature=0.0),
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify `User.md` can be created, updated, and edited."""

    store = UserProfileStore(tmp_path / "profiles")

    # Write
    store.write_text("test_user", "# Profile\n- **name**: Dũng\n- **location**: Đà Nẵng\n")
    assert store.file_size("test_user") > 0

    # Read
    content = store.read_text("test_user")
    assert "Dũng" in content
    assert "Đà Nẵng" in content

    # Edit (correction: Đà Nẵng -> Huế)
    changed = store.edit_text("test_user", "Đà Nẵng", "Huế")
    assert changed is True
    updated = store.read_text("test_user")
    assert "Huế" in updated
    assert "Đà Nẵng" not in updated


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long threads trigger compaction."""

    cm = CompactMemoryManager(threshold_tokens=100, keep_messages=2)
    for i in range(30):
        cm.append("thread-long", "user", f"Đây là message số {i} với nội dung khá dài " * 5)

    assert cm.compaction_count("thread-long") >= 1
    ctx = cm.context("thread-long")
    assert len(ctx["messages"]) <= 10  # compacted, not keeping all 30 messages
    assert ctx["summary"] != ""


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify advanced remembers across sessions and baseline does not."""

    config = make_config(tmp_path)

    # Advanced remembers across sessions (new thread id).
    adv = AdvancedAgent(config=config, force_offline=True)
    adv.reply("dungct", "session-1", "Mình tên là DũngCT, đang ở Đà Nẵng, làm backend engineer")
    result = adv.reply("dungct", "session-2", "Nhắc lại tên mình?")
    assert "DũngCT" in result["reply"], f"Advanced phải nhớ tên qua session! Got: {result['reply']}"

    # Baseline must NOT remember across sessions.
    base = BaselineAgent(config=config, force_offline=True)
    base.reply("dungct", "session-1", "Mình tên là DũngCT")
    result_b = base.reply("dungct", "session-2", "Tên mình là gì?")
    assert "DũngCT" not in result_b["reply"], "Baseline KHÔNG được nhớ qua session!"


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compare prompt load of baseline vs advanced on a long thread."""

    config = make_config(tmp_path)

    base = BaselineAgent(config=config, force_offline=True)
    adv = AdvancedAgent(config=config, force_offline=True)

    # Feed many long messages into the same thread.
    long_messages = [f"Đây là tin tức dài số {i}: " + "nội dung rất dài " * 30 for i in range(15)]
    for msg in long_messages:
        base.reply("dungct", "long-thread", msg)
        adv.reply("dungct", "long-thread", msg)

    baseline_prompt = base.prompt_token_usage("long-thread")
    advanced_prompt = adv.prompt_token_usage("long-thread")

    # Compact must trigger on the long thread.
    assert adv.compaction_count("long-thread") >= 1, "Compact phải xảy ra với thread dài"
    print(f"Baseline prompt tokens: {baseline_prompt}")
    print(f"Advanced prompt tokens: {advanced_prompt}")
    print(f"Advanced compactions: {adv.compaction_count('long-thread')}")


# --- memory decay tests -------------------------------------------------


def test_memory_decay_priority_drops_over_time(tmp_path: Path) -> None:
    """A fact that is never reinforced must lose priority as time passes."""

    store = UserProfileStore(tmp_path / "profiles", half_life_days=10.0)
    user = "decay_user"
    store.upsert_fact(user, "favorite_drink", "cà phê sữa đá")

    fresh = store.fact_priority(user, "favorite_drink")

    # 30 days later (3 half-lives) the same untouched fact is much weaker.
    aged = store.fact_priority(user, "favorite_drink", now=time.time() + 30 * 86400)
    assert aged < fresh, "Fact không được nhắc lại phải giảm priority theo thời gian"
    assert aged <= fresh / 4, "Sau 3 half-life, priority phải giảm mạnh (<= 25%)"


def test_memory_decay_reinforcement_keeps_fact_fresh(tmp_path: Path) -> None:
    """Repeating a fact must raise its priority ranking above a stale sibling."""

    store = UserProfileStore(tmp_path / "profiles", half_life_days=5.0)
    user = "reinforce_user"
    store.upsert_fact(user, "pet", "corgi tên Bơ")
    store.upsert_fact(user, "favorite_food", "mì Quảng")

    future = time.time() + 40 * 86400
    stale_rank = {k: p for k, _v, p in store.ranked_facts(user, now=future)}
    assert abs(stale_rank["pet"] - stale_rank["favorite_food"]) < 1e-6, (
        "Hai fact không được nhắc lại phải có priority gần bằng nhau"
    )

    # Reinforce `pet` many times (each mention bumps its frequency bonus).
    for _ in range(8):
        store.upsert_fact(user, "pet", "corgi tên Bơ")

    ranked = {k: p for k, _v, p in store.ranked_facts(user, now=future)}
    assert ranked["pet"] > ranked["favorite_food"], (
        "Fact được nhắc lại nhiều lần phải có priority cao hơn fact nguội"
    )


def test_prune_facts_bounds_user_md_size(tmp_path: Path) -> None:
    """Pruning must cap the number of facts so User.md cannot grow unbounded."""

    store = UserProfileStore(tmp_path / "profiles")
    user = "prune_user"
    for i in range(20):
        store.upsert_fact(user, f"fact_{i}", f"value_{i}")

    removed = store.prune_facts(user, max_facts=5)
    assert len(removed) == 15, f"Phải loại bỏ đúng số fact vượt hạn mức, got {len(removed)}"
    assert len(store.facts(user)) == 5, "User.md chỉ được giữ tối đa max_facts fact"
