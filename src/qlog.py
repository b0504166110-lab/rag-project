"""יומן פנימי: שורת JSON לכל שאלה בקובץ logs/queries.jsonl (התיקייה אינה עולה ל-git).

אין ביומן מפתחות או הודעות שגיאה גולמיות, כי הודעות שגיאה של ספקים עלולות לכלול חלק ממפתח.
"""
import json
from datetime import datetime

from config import EMBED_MODEL, LLM_MODEL, LOGS_DIR

LOG_FILE = LOGS_DIR / "queries.jsonl"


def _append(record: dict) -> None:
    LOGS_DIR.mkdir(exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_trace(tr, source: str = "app", embed_model: str | None = None) -> None:
    _append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "models": {"llm": LLM_MODEL, "embedding": embed_model or EMBED_MODEL},
        "question": tr.question,
        "filters": tr.filters,
        "retried": tr.retried,
        "retrieved": tr.retrieved,  # בסדר הדירוג הסופי, עם ציון אחזור וציון דירוג
        "selected": [{"file": s["file"], "section": s["section"]} for s in tr.selected],
        "rerank_failed": tr.rerank_failed,
        "answer": tr.answer,
        "is_refusal": tr.is_refusal,
        "seconds": tr.seconds,
        "tokens": tr.tokens,
    })


def log_error(source: str, error: Exception) -> None:
    _append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "error_type": type(error).__name__,
    })


def read_log() -> list[dict]:
    if not LOG_FILE.exists():
        return []
    return [json.loads(line) for line in LOG_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
