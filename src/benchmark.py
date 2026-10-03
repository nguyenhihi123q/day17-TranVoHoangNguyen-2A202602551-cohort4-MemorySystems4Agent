from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read JSON conversations from disk."""

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    """Fraction of expected facts that appear in the answer (0.0 - 1.0)."""

    if not expected:
        return 1.0
    answer_lower = answer.lower()
    found = sum(1 for e in expected if e.lower() in answer_lower)
    return found / len(expected)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight quality score for offline mode.

    Blends recall with a small length bonus so empty/tiny answers score low.
    """

    if not answer.strip():
        return 0.0
    recall = recall_points(answer, expected)
    length_ok = 1.0 if len(answer) > 20 else 0.5
    return round(recall * 0.7 + length_ok * 0.3, 2)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Evaluate one agent over many conversations.

    1. Feed all turns to the agent.
    2. Track `agent tokens only`.
    3. Track `prompt tokens processed`.
    4. Ask recall questions in a fresh thread.
    5. Compute average recall and quality.
    6. Record memory file growth and compaction count.
    """

    total_agent_tokens = 0
    total_prompt_tokens = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []
    total_compactions = 0
    memory_growth = 0
    last_user_id = ""

    for conv in conversations:
        user_id = conv["user_id"]
        last_user_id = user_id
        thread_id = conv["id"]

        # Feed every turn of the conversation into the same thread.
        for turn in conv.get("turns", []):
            agent.reply(user_id, thread_id, turn)

        total_agent_tokens += agent.token_usage(thread_id)
        total_prompt_tokens += agent.prompt_token_usage(thread_id)
        total_compactions += agent.compaction_count(thread_id)

        # Ask recall questions in a brand-new thread (cross-session recall).
        for question in conv.get("recall_questions", []):
            recall_thread = f"{thread_id}-recall"
            result = agent.reply(user_id, recall_thread, question["question"])
            answer = result["reply"]
            expected = question.get("expected_contains", [])
            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(heuristic_quality(answer, expected))

    if hasattr(agent, "memory_file_size") and last_user_id:
        memory_growth = agent.memory_file_size(last_user_id)

    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=round(avg_recall, 2),
        response_quality=round(avg_quality, 2),
        memory_growth_bytes=memory_growth,
        compactions=total_compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Print a markdown table or tabulated output."""

    try:
        from tabulate import tabulate
    except ImportError:
        tabulate = None

    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    table = [
        [
            r.agent_name,
            r.agent_tokens_only,
            r.prompt_tokens_processed,
            f"{r.recall_score:.0%}",
            f"{r.response_quality:.0%}",
            r.memory_growth_bytes,
            r.compactions,
        ]
        for r in rows
    ]

    if tabulate is not None:
        return tabulate(table, headers=headers, tablefmt="github")

    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in table:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def reset_state(state_dir: Path) -> None:
    """Wipe persisted memory so every benchmark run starts from a clean slate.

    Without this, `User.md` + `_meta.json` from a previous run leak into the next
    one (extra facts / mention counters), making the columns drift run-to-run.
    """

    profiles = Path(state_dir) / "profiles"
    if profiles.exists():
        import shutil

        shutil.rmtree(profiles, ignore_errors=True)


def main() -> None:
    """Run both benchmark suites and print the comparison tables.

    - Standard benchmark from `data/conversations.json`
    - Long-context stress benchmark from `data/advanced_long_context.json`
    """

    import sys
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

    config = load_config(Path(__file__).resolve().parent.parent)

    # Deterministic, reproducible numbers: start from a clean memory state.
    reset_state(config.state_dir)

    # === Standard Benchmark ===
    convs = load_conversations(config.data_dir / "conversations.json")
    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)

    row_b = run_agent_benchmark("Baseline", baseline, convs, config)
    row_a = run_agent_benchmark("Advanced", advanced, convs, config)

    print("\n=== Standard Benchmark ===\n")
    print(format_rows([row_b, row_a]))

    # === Long-Context Stress Benchmark ===
    stress = load_conversations(config.data_dir / "advanced_long_context.json")
    baseline2 = BaselineAgent(config=config, force_offline=True)
    advanced2 = AdvancedAgent(config=config, force_offline=True)

    row_b2 = run_agent_benchmark("Baseline", baseline2, stress, config)
    row_a2 = run_agent_benchmark("Advanced", advanced2, stress, config)

    print("\n=== Long-Context Stress Benchmark ===\n")
    print(format_rows([row_b2, row_a2]))

    # === Analysis ===
    print("\n=== Phân tích kết quả ===\n")
    print("1. Advanced recall tốt hơn Baseline vì User.md lưu facts bền vững qua session mới,")
    print("   trong khi Baseline chỉ nhớ trong cùng thread_id nên quên ngay khi đổi thread.")
    print("2. Ở hội thoại ngắn, Advanced có thể tốn agent tokens hơn Baseline do overhead")
    print("   đọc/ghi User.md và inject profile + summary vào mỗi lượt.")
    print("3. Ở hội thoại rất dài, compact memory giúp Advanced chặn đà tăng của")
    print("   `Prompt tokens processed` (nén lịch sử cũ thành summary) trong khi Baseline")
    print("   phải mang theo toàn bộ lịch sử mỗi lượt -> prompt cost bùng nổ.")
    print("4. User.md tăng trưởng theo số fact ổn định -> rủi ro phình file, lưu sai fact,")
    print("   hoặc trùng lặp; cần guardrail (confidence threshold, conflict handling).")
    print(f"5. Memory growth: {row_a.memory_growth_bytes} bytes (standard), "
          f"{row_a2.memory_growth_bytes} bytes (stress).")
    print("6. Bonus đã implement: confidence threshold (bỏ câu hỏi thuần + nhiễu),")
    print("   conflict handling (upsert fact mới nhất, correction đè fact cũ), và")
    print("   memory decay (recency half-life + tần suất nhắc lại + prune fact nguội).")
    print("   Decay giúp inject fact mới/tần suất cao lên đầu prompt và chặn User.md")
    print("   phình to, nhưng nếu half-life quá ngắn có thể quên fact cũ vẫn còn đúng.")


if __name__ == "__main__":
    main()
