"""סט רגרסיה קבוע לשאלות מחיר: כל שאלה מורצת כמה פעמים, ומדווחים נכון / סירוב / מחיר שגוי / ללא מחיר.
  python scripts/price_regression.py --runs 5 --label שם
  python scripts/price_regression.py --questions eval/questions_price_regression.json --runs 5

סיווג: סירוב לפי תשובת הסירוב; נכון אם מופיע המחיר הצפוי; שגוי אם מופיע מחיר (מספר לפני ₪) אחר בלי הצפוי;
"ללא מחיר" אם התשובה אינה סירוב ואין בה מחיר בכלל. תשובה שכוללת גם את הצפוי וגם מחיר שגוי מסומנת "מעורב" לבדיקה ידנית.
"""
import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import PRICES, LLM_MODEL, RERANK_GROUP, RETRIEVE_K  # noqa: E402
from models import MissingApiKey  # noqa: E402
from rag import build_pipeline  # noqa: E402

EST_TOKENS_PER_ASK = 3_400  # הערכה גסה להרצה אחת ב-K=10; ב-K=20 כ-5,900
PRICE_RE = re.compile(r"(\d[\d,.]*)\s*₪")


def classify(tr, q) -> str:
    if tr.is_refusal:
        return "refusal"
    prices = {m.replace(",", "") for m in PRICE_RE.findall(tr.answer)}
    has_expected = any(e in prices for e in q["expected_values"])
    has_wrong = bool(prices - set(q["expected_values"]))
    if has_expected and has_wrong:
        return "mixed"
    if has_expected:
        return "correct"
    if has_wrong:
        return "wrong"
    return "no_price"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(ROOT / "eval" / "questions_price_regression.json"))
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    n = len(questions) * args.runs
    per_ask = EST_TOKENS_PER_ASK * (2 if RETRIEVE_K > 10 else 1)
    est = n * per_ask * PRICES[LLM_MODEL]["input"] / 1e6 * 1.2
    print(f"הערכת עלות: כ-${est:.3f} ({n} הרצות). K={RETRIEVE_K}, קבוצת דירוג={RERANK_GROUP}")
    try:
        pipe = build_pipeline()
    except MissingApiKey as e:
        print(e)
        return 1
    out, tokens = [], 0
    for q in questions:
        counts = {"correct": 0, "refusal": 0, "wrong": 0, "no_price": 0, "mixed": 0}
        answers = []
        for _ in range(args.runs):
            tr = pipe.ask(q["question"])
            c = classify(tr, q)
            counts[c] += 1
            tokens += tr.tokens["total_llm"]
            answers.append({"class": c, "answer": tr.answer, "sent": [s["file"] for s in tr.selected]})
        out.append({"id": q["id"], "question": q["question"], "counts": counts, "answers": answers})
        print(f"\n{q['id']} {q['question']}  ->  {counts}")
        for a in answers:
            print(f"   [{a['class']}] {a['answer']}  | נשלחו: {a['sent']}")
    tot = {k: sum(o["counts"][k] for o in out) for k in ("correct", "refusal", "wrong", "no_price", "mixed")}
    cost = tokens * PRICES[LLM_MODEL]["input"] / 1e6  # הערכה תחתונה: קלט בלבד
    print(f"\nסך הכל: {tot} | טוקנים: {tokens} | עלות קלט משוערת ${cost:.4f}")
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    report = {"timestamp": stamp, "label": args.label,
              "config": {"retrieve_k": RETRIEVE_K, "rerank_group": RERANK_GROUP, "runs": args.runs},
              "totals": tot, "tokens": tokens, "results": out}
    (ROOT / "reports").mkdir(exist_ok=True)
    path = ROOT / "reports" / f"price_{stamp}{'_' + args.label if args.label else ''}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("נשמר:", path.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
