# Phân tích Memory System (Day 17, Track 3)

Tài liệu này giải thích **vì sao** hệ thống memory hoạt động như kết quả benchmark,
bằng số liệu đo được thật từ `python src/benchmark.py` (đã bật reset state nên
chạy lại luôn ra cùng kết quả).

## 1. Ba tầng memory được tách bạch trong code

| Tầng | Vai trò | Nơi cài đặt | Vòng đời |
|---|---|---|---|
| **Short-term** | Nhớ trong cùng `thread_id` | `CompactMemoryManager.messages` | Sống theo thread, mất khi đổi thread |
| **Persistent** | Hồ sơ fact ổn định của user | `UserProfileStore` → `state/profiles/<user>/User.md` | Sống qua mọi session, đọc từ đĩa |
| **Compact** | Nén lịch sử cũ của thread dài | `CompactMemoryManager` (summary + keep-N) | Sống theo thread, tự kích hoạt theo ngưỡng token |

`BaselineAgent` **chỉ** có tầng short-term (dict theo `thread_id`), nên sang thread
mới là quên sạch. `AdvancedAgent` có đủ cả ba tầng.

## 2. Baseline quên dài hạn → Advanced nhớ nhờ User.md

Số liệu (Standard Benchmark, 10 hội thoại × ~10 lượt, 15 câu hỏi recall ở thread mới):

| Agent | Cross-session recall | Response quality |
|---|---|---|
| Baseline | **10%** | 37% |
| Advanced | **89%** | 92% |

**Vì sao:** recall questions được hỏi ở `thread_id` hoàn toàn mới. Baseline tra
`self.sessions[thread_id]` → rỗng → không có fact nào để trả lời. Advanced tra
`User.md` trên đĩa → fact vẫn còn. Đây chính là "luồng logic" rubric yêu cầu:
baseline không nhớ dài hạn, advanced thêm `User.md` nên recall tăng vọt.

## 3. Hội thoại ngắn: compact **không** phải lúc nào cũng thắng

Ở Standard Benchmark, `Advanced` tốn **nhiều token hơn** Baseline:

- Agent tokens only: **5409** vs 3347
- Prompt tokens processed: **31050** vs 16596

**Vì sao:** mỗi lượt Advanced phải đọc `User.md`, dựng summary, rồi nhét toàn bộ
profile + summary + recent messages vào prompt. Đó là **overhead cố định** của
persistent + compact memory. Khi hội thoại còn ngắn, ngưỡng compact (1500 token)
chưa kích hoạt (`Compactions = 0`), nên Advanced chỉ trả giá mà chưa thu lợi.
→ Memory mạnh hơn luôn đi kèm chi phí, không miễn phí.

## 4. Hội thoại rất dài: compact kéo chi phí ngữ cảnh xuống

Số liệu (Long-Context Stress Benchmark, 1 hội thoại 16 lượt rất dài):

| Agent | Agent tokens only | Prompt tokens processed | Compactions |
|---|---|---|---|
| Baseline | 2784 | **233966** | 0 |
| Advanced | 2884 | **16759** | 3 |

**Vì sao:** Baseline giữ *toàn bộ* lịch sử trong `session.messages` và mỗi lượt
tính `prompt_ctx = sum(tokens của mọi message)` → tổng prompt tăng theo cấp số
cộng, bùng nổ lên ~234k token. Advanced nén lịch sử cũ thành `summary` và chỉ giữ
6 message gần nhất (`keep_messages`), nên prompt gần như **đi ngang** → chỉ ~17k
token, **giảm ~93%**.

Đây là điểm cốt lõi: **compact memory chủ yếu tối ưu `Prompt tokens processed`**,
không phải `Agent tokens only`. Agent tokens của hai bên gần bằng nhau (~2.8k) vì
nội dung sinh ra ở mỗi lượt là tương đương; thứ compact cắt là *ngữ cảnh phải
mang theo mỗi lần gọi model*.

## 5. Rủi ro của memory file & guardrail đã thêm

`User.md` tăng trưởng theo số fact, nên có 3 rủi ro thật:

1. **Phình file**: nhiều fact dài → prompt persistent phình, lại làm tăng cost.
2. **Lưu sai fact**: extractor bắt nhầm nhiễu (ví dụ "Hà Nội" chỉ là nơi đi họp,
   "product manager" chỉ là câu đùa) → agent trả lời sai tự tin.
3. **Xung đột fact**: correction (backend → MLOps, Đà Nẵng → Huế) nếu không xử lý
   sẽ khiến file giữ đồng thời fact cũ và mới.

Guardrail đã cài:

- **Confidence threshold**: `_is_pure_question()` bỏ câu hỏi thuần; `_FILLER`,
  `_QUESTION_WORDS`, `_LOC_NOISE`, `_TECH_HINT_RE` chặn giá trị rác.
- **Conflict handling**: `upsert_fact()` ghi đè theo key; `_OLD_MARKERS` ("không
  còn", "đừng") loại mention cũ; `_NEW_MARKERS` ("chuyển sang") ưu tiên mention mới.
- **Memory decay** (xem mục 6): fact nguội bị hạ ưu tiên và prune.

## 6. Bonus: Memory decay giải quyết gì, cải thiện gì, rủi ro gì

**Cơ chế** (`UserProfileStore`):

- Mỗi fact có `mentions` (số lần được nhắc lại) và `updated_at` (lần cuối đổi giá
  trị), lưu trong sidecar `state/profiles/<user>/_meta.json`.
- `priority = 0.5 ** (age_days / half_life) * (1 + 0.15 * log2(mentions))` — decay
  theo thời gian, cộng bonus theo tần suất.
- `ranked_facts()` inject fact theo priority: fact mới / hay được nhắc lên đầu prompt.
- `prune_facts()` bỏ fact vượt `max_facts` hoặc priority < 0.15 sau mỗi lần ghi.

**Giải quyết vấn đề gì:** fact cũ, ít dùng, có thể đã sai không còn chiếm chỗ trong
prompt; `User.md` không phình vô hạn.

**Cải thiện đo được** (đo thật, half-life = 30 ngày):

| Tình huống | Priority |
|---|---|
| Fact không nhắc lại, sau 30 ngày | 0.5000 |
| Fact không nhắc lại, sau 90 ngày | 0.1250 |
| Fact được nhắc lại 20 lần, sau 30 ngày | 0.8294 |
| Fact được nhắc lại 20 lần, sau 90 ngày | 0.2207 |

→ Cùng độ "cũ", fact được reinforcement có priority **cao hơn 66%** (0.83 vs 0.50).
Về tác động lên hệ thống: prune + decay làm `Memory growth` ở stress test giảm từ
**4005 → 215 bytes** và `Compactions` từ 11 → 3, **mà recall vẫn giữ 100%** — file
gọn hơn, prompt persistent nhẹ hơn, chất lượng không đổi.

**Rủi ro mới:** nếu `half_life` quá ngắn, một fact cũ *vẫn còn đúng* (ví dụ "quê ở
Huế") có thể bị decay/prune nhầm → mất thông tin thật. Vì vậy default để half-life
khá dài (30 ngày) và `min_priority` thấp (0.15); đây là **trade-off cấu hình**, không
phải hằng số đúng cho mọi domain.

