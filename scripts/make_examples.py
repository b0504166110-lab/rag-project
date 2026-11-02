"""מפיק את docs/examples.md מהיומן הפנימי (logs/queries.jsonl).

לוקח רק רשומות שנרשמו עם התווית examples, כלומר:
  python src/ask.py --source examples "שאלה 1" "שאלה 2" ...
ואז:
  python scripts/make_examples.py
אם אותה שאלה נשאלה כמה פעמים, נלקחת ההרצה האחרונה. כל הנתונים במסמך מגיעים מהיומן, בלי עריכה ידנית.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qlog import read_log  # noqa: E402

OUT = ROOT / "docs" / "examples.md"
TOP_SHOWN = 5


def main() -> int:
    entries = [r for r in read_log() if r.get("source") == "examples" and "question" in r]
    latest: dict[str, dict] = {}
    for r in entries:
        latest[r["question"]] = r  # האחרונה גוברת
    if not latest:
        print("אין ביומן רשומות עם התווית examples. ראו את ההוראות בראש הסקריפט.")
        return 1
    lines = [
        "# דוגמאות לשאלות ותשובות",
        "",
        "המסמך הופק אוטומטית מהיומן הפנימי בעזרת `scripts/make_examples.py`, בלי עריכה ידנית. "
        "השאלות הורצו בפועל על המערכת הסופית. "
        "בכל דוגמה מופיעים הקטעים שנשלחו למודל התשובה, ומתחתיהם הקטעים שנשלפו בראש הדירוג, עם ציון האחזור (`vector`) "
        "וציון הדירוג של המודל (`rerank`, 0 עד 10). הממשק עצמו אינו מציג את המקורות.",
        "",
    ]
    for n, r in enumerate(latest.values(), 1):
        refusal = "כן" if r.get("is_refusal") else "לא"
        lines += [
            f"## דוגמה {n}",
            "",
            f"**שאלה:** {r['question']}",
            "",
            f"**תשובה:** {r['answer']}",
            "",
            f"- סירוב (אין מידע במאגר): {refusal}",
            f"- חיפוש חוזר בלי סינון: {'כן' if r.get('retried') else 'לא'}",
            f"- משחק שזוהה בשאלה: {', '.join(r.get('filters', {}).get('games', [])) or 'אין'}",
            f"- זמן: {r['seconds']} שניות, טוקנים ב-LLM: {r['tokens'].get('total_llm')}",
            f"- הרצה: {r['timestamp']}, מודלים: {r['models']['llm']} / {r['models']['embedding']}",
            "",
            "**קטעים שנשלחו למודל התשובה:**",
            "",
        ]
        if r["selected"]:
            lines += [f"{i}. `{s['file']}`, סעיפים: {s['section']}" for i, s in enumerate(r["selected"], 1)]
        else:
            lines.append("לא נשלחו קטעים.")
        lines += ["", f"**הקטעים בראש הדירוג (עד {TOP_SHOWN}):**", "", "| # | קובץ | סוג מסמך | vector | rerank |", "|---|---|---|---|---|"]
        for i, t in enumerate(r["retrieved"][:TOP_SHOWN], 1):
            lines.append(f"| {i} | `{t['file']}` | {t['doc_type']} | {t['vector_score']} | {t['rerank_score']} |")
        lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"נכתבו {len(latest)} דוגמאות אל {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
