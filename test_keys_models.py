"""Test: kiểm tra tất cả API keys và models hoạt động ổn không.

Chạy:
    uv run python test_keys_models.py
    # hoặc
    python test_keys_models.py

Sẽ lần lượt thử từng key × từng model, báo kết quả.
"""

import json
import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_keys")

# ── Load config ──────────────────────────────────────────────────────────
from dotenv import load_dotenv

load_dotenv()

from app.config import (
    GEMINI_API_KEY,
    GEMINI_API_KEY_2,
    GEMINI_API_KEY_3,
    LLM_API_KEYS,
    LLM_MODELS,
)

# ── Test helpers ─────────────────────────────────────────────────────────

def try_model(api_key: str, model: str, key_label: str) -> dict:
    """Thử 1 model với 1 key, trả về kết quả."""
    from google import genai

    client = genai.Client(api_key=api_key)
    start = time.time()
    try:
        response = client.models.generate_content(
            model=model,
            contents="Trả lời: OK (chỉ 1 từ)",
        )
        elapsed = time.time() - start
        if response and hasattr(response, "text") and response.text:
            usage = getattr(response, "usage_metadata", None)
            tokens = (
                f"prompt={getattr(usage, 'prompt_token_count', '?')} "
                f"candidate={getattr(usage, 'candidates_token_count', '?')}"
            ) if usage else "N/A"
            return {
                "status": "✅ OK",
                "time": f"{elapsed:.1f}s",
                "tokens": tokens,
            }
        return {"status": "⚠️ No text", "time": f"{elapsed:.1f}s", "tokens": "N/A"}
    except Exception as e:
        elapsed = time.time() - start
        err = str(e)
        if "429" in err or "quota" in err.lower():
            return {"status": "❌ Hết quota", "time": f"{elapsed:.1f}s", "tokens": "-"}
        return {"status": f"❌ Lỗi", "time": f"{elapsed:.1f}s", "tokens": err[:80]}


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print()
    print("=" * 70)
    print("  🔑 TEST GEMINI API KEYS + MODELS")
    print("=" * 70)
    print()

    # ── Thông tin keys ──────────────────────────────────────────────────
    keys_info = [
        ("Key 1 (GEMINI_API_KEY)", GEMINI_API_KEY),
        ("Key 2 (GEMINI_API_KEY_2)", GEMINI_API_KEY_2),
        ("Key 3 (GEMINI_API_KEY_3)", GEMINI_API_KEY_3),
    ]

    print("📋 Danh sách keys:")
    for label, key in keys_info:
        if key:
            masked = key[:12] + "..." + key[-4:]
            print(f"   {label}: {masked}")
        else:
            print(f"   {label}: ⚠️  Không có")
    print()

    print("📋 Danh sách models:")
    for i, m in enumerate(LLM_MODELS, 1):
        print(f"   Model {i}: {m}")
    print()

    # ── Test từng key × model ───────────────────────────────────────────
    print("-" * 70)
    print("  🔬 BẮT ĐẦU TEST...")
    print("-" * 70)
    print()

    all_ok = True
    results = []

    for key_label, api_key in keys_info:
        if not api_key:
            results.append((key_label, [{"status": "⏭️ Bỏ qua (no key)", "time": "-", "tokens": "-"}]))
            continue

        row_results = []
        for model in LLM_MODELS:
            logger.info("  Đang test %s × %s ...", key_label.split()[0], model)
            result = try_model(api_key, model, key_label)
            row_results.append(result)
            if "❌" in result["status"]:
                all_ok = False

        results.append((key_label, row_results))

    # ── Bảng kết quả ────────────────────────────────────────────────────
    print()
    print("-" * 70)
    print("  📊 KẾT QUẢ")
    print("-" * 70)

    # Header
    header = f"{'Key':<20}"
    for i, m in enumerate(LLM_MODELS, 1):
        short = m.replace("models/gemini-", "").replace("-preview", "")
        header += f" | {short:<22}"
    print(header)
    print("-" * len(header))

    for key_label, row_results in results:
        key_short = key_label.split()[0] + " " + key_label.split()[1]
        line = f"{key_short:<20}"
        for r in row_results:
            line += f" | {r['status'] + ' ' + r['time']:<22}"
        print(line)

    # ── Tổng kết ────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    if all_ok:
        print("  🎯 TẤT CẢ OK — tất cả keys và models đều hoạt động!")
    else:
        print("  ⚠️  Một số key/model bị lỗi. Xem chi tiết ở trên.")
    print("=" * 70)
    print()

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
