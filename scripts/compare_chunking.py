"""השוואת שיטות חלוקה מתוך דוחות הערכה. שימוש:
  python scripts/compare_chunking.py

קורא את reports/eval_*_chunk_<שיטה>_r<חזרה>.json ומדפיס טבלאות Markdown לפי קבוצות שאלות,
כשכל ערך הוא ממוצע על חזרות ההרצה (המספר בסוגריים הוא ערך לכל חזרה).
"""
import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METHODS = ["sentence", "markdown"]
GROUPS = {
    "10 הראשונות": range(1, 11),
    "20 הנוספות": range(101, 121),
    "30 יחד": list(range(1, 11)) + list(range(101, 121)),
    "שאלת החלקים החסרים": [401],
    "5 שאלות לקוחה": range(301, 306),
    "הכל (36)": None,
}


def load(method):
    files = sorted(glob.glob(str(ROOT / "reports" / f"eval_*_chunk_{method}_r*.json")))
    return [json.loads(Path(f).read_text(encoding="utf-8")) for f in files]


def metrics(results):
    ans = [r for r in results if r["answerable"]]
    un = [r for r in results if not r["answerable"]]

    def m(xs):
        xs = list(xs)
        return sum(xs) / len(xs) if xs else None

    return {
        "R@3": m(r["recall_at_3"] for r in ans), "R@10": m(r["recall_at_10"] for r in ans),
        "MRR": m(r["rr"] for r in ans), "נכונות": m(r["correctness"] for r in ans),
        "סירוב שגוי": sum(r["is_refusal"] for r in ans), "n_ans": len(ans),
        "סירוב נכון": sum(r["is_refusal"] for r in un), "n_un": len(un),
        "נאמנות": m(r["faithfulness"] for r in results if not r["is_refusal"]),
    }


def sel(report, ids):
    return [r for r in report["results"] if ids is None or r["id"] in set(ids)]


def fmt(vals, pct=True):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    avg = sum(vals) / len(vals)
    each = ", ".join(f"{v * 100:.0f}%" if pct else f"{v:.2f}" for v in vals)
    return (f"{avg * 100:.0f}%" if pct else f"{avg:.2f}") + f" ({each})"


def main() -> int:
    data = {m: load(m) for m in METHODS}
    if not all(data.values()):
        print("חסרים דוחות. יש להריץ evaluate.py עם --label chunk_<שיטה>_r<חזרה>.")
        return 1
    print(f"חזרות לכל שיטה: " + ", ".join(f"{m}={len(v)}" for m, v in data.items()), "\n")
    for gname, ids in GROUPS.items():
        print(f"### {gname}\n")
        print("| מדד | " + " | ".join(METHODS) + " |")
        print("|---|" + "---|" * len(METHODS))
        per = {m: [metrics(sel(rep, ids)) for rep in reps] for m, reps in data.items()}
        for key, pct in (("R@3", True), ("R@10", True), ("MRR", False), ("נכונות", True), ("נאמנות", True)):
            print(f"| {key} | " + " | ".join(fmt([x[key] for x in per[m]], pct) for m in METHODS) + " |")
        n_ans = per[METHODS[0]][0]["n_ans"]
        n_un = per[METHODS[0]][0]["n_un"]
        print(f"| סירובים שגויים (מתוך {n_ans}) | " + " | ".join(", ".join(str(x["סירוב שגוי"]) for x in per[m]) for m in METHODS) + " |")
        print(f"| סירובים נכונים (מתוך {n_un}) | " + " | ".join(", ".join(str(x["סירוב נכון"]) for x in per[m]) for m in METHODS) + " |")
        print()
    print("### זמן וטוקנים (כל 36 השאלות)\n")
    print("| מדד | " + " | ".join(METHODS) + " |")
    print("|---|" + "---|" * len(METHODS))
    for key, label in (("avg_seconds", "זמן ממוצע לשאלה (שניות)"), ("avg_pipeline_llm_tokens_per_question", "טוקנים ב-LLM לשאלה")):
        print(f"| {label} | " + " | ".join(
            ", ".join(f"{rep['summary'][key]:.2f}" if key == "avg_seconds" else f"{rep['summary'][key]:.0f}" for rep in data[m])
            for m in METHODS) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
