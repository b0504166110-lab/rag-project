"""השוואת תצורות תיקון מתוך דוחות הערכה. שימוש:
  python scripts/compare_fix.py fix_base fix_k20 fix_v2 fix_both

קורא reports/eval_*_<תווית>.json. מדפיס: (א) שאלות הלקוחה החדשות בנפרד, (ב) 36 השאלות הקיימות,
(ג) שאלות מפתח (108, 114, 304, 401), (ד) בדיקת שיוך מחיר לשאלות המחיר, (ה) סירוב בשאלות ללא תשובה.
"""
import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NEW = range(501, 507)
KEY = (108, 114, 304, 401)


def load(label):
    f = sorted(glob.glob(str(ROOT / "reports" / f"eval_*_{label}.json")))[-1]
    return json.loads(Path(f).read_text(encoding="utf-8"))


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def pct(v):
    return f"{v * 100:.0f}%"


def group_metrics(res):
    ans = [r for r in res if r["answerable"]]
    un = [r for r in res if not r["answerable"]]
    return {
        "נכונות": pct(mean(r["correctness"] for r in ans)) if ans else "-",
        "Recall@3": pct(mean(r["recall_at_3"] for r in ans)) if ans else "-",
        "כיסוי בקטעים שנשלחו": pct(mean(r["final_recall"] for r in ans)) if ans else "-",
        "סירובים שגויים": f"{sum(r['is_refusal'] for r in ans)}/{len(ans)}",
        "סירובים נכונים": f"{sum(r['is_refusal'] for r in un)}/{len(un)}" if un else "-",
        "קטעים שנשלחו (ממוצע)": f"{mean(len(r['selected']) for r in res):.2f}",
        "קטעים שאושרו (ממוצע)": f"{mean(sum(1 for t in r['retrieved'] if t['rerank_score'] >= 5) for r in res):.2f}",
    }


def nums(text):
    return set(re.findall(r"\d[\d,]*", text.replace(",", "")))


def main() -> int:
    labels = sys.argv[1:]
    if not labels:
        print(__doc__)
        return 1
    reps = {l: load(l) for l in labels}
    qfile = {q["id"]: q for q in json.loads((ROOT / "eval" / "questions_fix42.json").read_text(encoding="utf-8"))}
    for title, pick in (("שאלות הלקוחה החדשות (501-506), הבדיקה הנקייה", lambda r: r["id"] in NEW),
                        ("36 השאלות הקיימות", lambda r: r["id"] not in NEW)):
        print(f"### {title}\n")
        print("| מדד | " + " | ".join(labels) + " |")
        print("|---|" + "---|" * len(labels))
        gm = {l: group_metrics([r for r in reps[l]["results"] if pick(r)]) for l in labels}
        for k in next(iter(gm.values())):
            print(f"| {k} | " + " | ".join(gm[l][k] for l in labels) + " |")
        print()
    print("### שאלות מפתח (נכונות; ס = סירוב)\n")
    print("| שאלה | " + " | ".join(labels) + " |")
    print("|---|" + "---|" * len(labels))
    for i in KEY + tuple(NEW):
        cells = []
        for l in labels:
            x = next(r for r in reps[l]["results"] if r["id"] == i)
            cells.append(f"{x['correctness']:g}" + (" ס" if x["is_refusal"] else ""))
        print(f"| {i} | " + " | ".join(cells) + " |")
    print("\n### שיוך מחיר למוצר הנכון (שאלות 501-503, 304)\n")
    for i in (501, 502, 503, 304):
        q = qfile[i]
        exp = q.get("expected_values") or (["129"] if i == 304 else [])
        wrong = q.get("wrong_values") or (["189", "119"] if i == 304 else [])
        print(f"**{i}. {q['question']}** (נכון: {', '.join(exp)}; מחירים שגויים אפשריים: {', '.join(wrong)})")
        for l in labels:
            x = next(r for r in reps[l]["results"] if r["id"] == i)
            n = nums(x["answer"])
            ok = all(e in n for e in exp)
            bad = [w for w in wrong if w in n]
            verdict = "נכון" if ok and not bad else ("נכון, ומוזכר גם מחיר אחר" if ok else "שגוי או חסר")
            print(f"- {l}: {verdict} | {x['answer'][:140]}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
