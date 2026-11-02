"""השוואה בין דוחות הערכה. שימוש: python scripts/compare_reports.py reports/a.json reports/b.json [...]
מדפיס טבלת Markdown, עמודה לכל דוח."""
import json
import sys
from pathlib import Path

ROWS = [
    ("מודל Embedding", lambda r: r["config"]["embedding"], None),
    ("Recall@3", lambda r: r["summary"]["recall_at_3"], "pct"),
    ("Recall@10", lambda r: r["summary"]["recall_at_10"], "pct"),
    ("MRR", lambda r: r["summary"]["mrr"], "num"),
    ("Faithfulness", lambda r: r["summary"]["faithfulness"], "pct"),
    ("Answer correctness (שאלות עם תשובה)", lambda r: r["summary"]["answer_correctness_answerable"], "pct"),
    ("סירובים נכונים (שאלות ללא תשובה)", lambda r: r["summary"]["correct_refusal_rate"], "pct"),
    ("סירובים שגויים (שאלות עם תשובה)", lambda r: r["summary"]["false_refusal_rate"], "pct"),
    ("שאלות עם סינון לפי שם משחק", lambda r: r["summary"].get("game_filters", "-"), None),
    ("חיפושים חוזרים", lambda r: r["summary"]["retries"], None),
    ("זמן ממוצע לשאלה (שניות)", lambda r: r["summary"]["avg_seconds"], "num"),
    ("טוקנים ב-LLM לשאלה (ממוצע)", lambda r: r["summary"]["avg_pipeline_llm_tokens_per_question"], "int"),
]


def fmt(v, kind):
    if kind == "pct":
        return f"{v * 100:.0f}%"
    if kind == "num":
        return f"{v:.2f}"
    if kind == "int":
        return f"{v:.0f}"
    return str(v)


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]]
    if len(paths) < 2:
        print(__doc__)
        return 1
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    names = [r["label"] or p.stem for r, p in zip(reports, paths)]
    print("| מדד | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for title, get, kind in ROWS:
        print(f"| {title} | " + " | ".join(fmt(get(r), kind) for r in reports) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
