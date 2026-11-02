"""הערכת המערכת על סט שאלות קבוע. הפעלה בפקודה אחת:
  python src/evaluate.py                                   הרצה עם מודל ה-Embedding שבהגדרות
  python src/evaluate.py --embed-model text-embedding-3-small --label small
  python src/evaluate.py --yes                             אישור להרצה שעלותה המשוערת מעל דולר

הדוח נשמר בתיקיית reports עם תאריך ושעה (JSON ו-Markdown), כדי להשוות בין הרצות.
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import chromadb

from config import (
    CHROMA_DIR, CHUNK_OVERLAP, CHUNK_SIZE, CHUNKING_METHOD, EMBED_MODEL,
    ANSWER_MODE, LLM_MODEL, PRICES, RERANK_GROUP, RERANK_MIN_RELEVANT, RERANK_TOP, RETRIEVE_K, ROOT, SEND_ONLY_RELEVANT, collection_name,
)
from llama_index.core.llms import ChatMessage
from models import MissingApiKey, get_llm
from rag import _usage, build_pipeline

QUESTIONS = ROOT / "eval" / "questions.json"
REPORTS = ROOT / "reports"

# הערכות גסות לשאלה אחת, לפי הרצות קודמות. משמשות להערכת עלות לפני הרצה בלבד.
EST_PIPELINE_INPUT = 3_600
EST_JUDGE_INPUT = 2 * 1_800
EST_OUTPUT = 140
EST_INDEX_TOKENS = 21_000  # בניית אינדקס חדש, אם האוסף עדיין לא קיים

FAITH_SYSTEM = (
    "אתה בודק נאמנות של תשובה לקטעי מקור. החזר JSON בלבד: {\"faithful\": 1 או 0}. "
    "faithful=1 רק אם כל עובדה בתשובה נתמכת במפורש בקטעים. אם יש בתשובה מידע שאינו בקטעים או סותר אותם, faithful=0."
)
CORRECT_SYSTEM = (
    "אתה בודק נכונות של תשובה מול תשובה נכונה ידועה. החזר JSON בלבד: {\"correct\": 1 או 0.5 או 0}. "
    "1 = התשובה כוללת את כל העובדות המהותיות בתשובה הנכונה ואינה סותרת אותה. "
    "0.5 = חלקית, חסרה עובדה מהותית אחת אך מה שנאמר נכון. 0 = שגויה, סותרת, או מסרבת לענות. "
    "הבדלי ניסוח אינם משנים את הציון."
)


def estimate_cost(n_questions: int, embed_model: str) -> tuple[float, str]:
    p = PRICES[LLM_MODEL]
    llm = n_questions * ((EST_PIPELINE_INPUT + EST_JUDGE_INPUT) * p["input"] + EST_OUTPUT * p["output"]) / 1e6
    exists = False
    try:
        exists = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(collection_name(embed_model)).count() > 0
    except Exception:
        pass
    emb = (0 if exists else EST_INDEX_TOKENS) * PRICES[embed_model]["input"] / 1e6
    note = "האינדקס קיים" if exists else "כולל בניית אינדקס חדש למודל זה"
    return llm + emb, note


def judge(llm, system: str, user: str) -> tuple[dict, int, int]:
    resp = llm.chat([ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)])
    try:
        data = json.loads(resp.message.content)
    except Exception:
        data = {}
    pi, po = _usage(resp)
    return data, pi, po


def relevant_ranks(retrieved: list[dict], expected: list[str]) -> list[int]:
    """דירוגים (מ-1) של הקטעים ששייכים למסמכים הצפויים."""
    return [i for i, r in enumerate(retrieved, 1) if r["file"] in expected]


def recall_at(retrieved: list[dict], expected: list[str], k: int) -> float:
    found = {r["file"] for r in retrieved[:k]} & set(expected)
    return len(found) / len(expected)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-model", default=EMBED_MODEL)
    ap.add_argument("--label", default="")
    ap.add_argument("--questions", default=str(QUESTIONS), help="קובץ שאלות (ברירת מחדל: הסט הראשי)")
    ap.add_argument("--yes", action="store_true", help="אישור להרצה שעלותה המשוערת מעל דולר")
    args = ap.parse_args()

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    est, note = estimate_cost(len(questions), args.embed_model)
    print(f"הערכת עלות להרצה: כ-${est:.3f} ({note}). {len(questions)} שאלות, מודל Embedding: {args.embed_model}")
    if est > 1.0 and not args.yes:
        print("העלות המשוערת עולה על דולר. נדרש אישור מפורש עם --yes.")
        return 2

    try:
        pipe = build_pipeline(embed_model_name=args.embed_model)
        judge_llm = get_llm(json_mode=True)
    except MissingApiKey as e:
        print(e)
        return 1

    results = []
    judge_in = judge_out = 0
    t_start = time.time()
    for q in questions:
        tr = pipe.ask(q["question"])
        ranks = relevant_ranks(tr.retrieved, q["expected_files"]) if q["answerable"] else []
        res = {
            "id": q["id"], "category": q["category"], "question": q["question"], "answerable": q["answerable"],
            "expected_type": q["doc_type"], "expected_files": q["expected_files"],
            "expected_answer": q["expected_answer"], "answer": tr.answer, "is_refusal": tr.is_refusal,
            "games": tr.games, "retried": tr.retried,
            "retrieved": [{k: r[k] for k in ("file", "section", "vector_score", "rerank_score")} for r in tr.retrieved],
            "selected": [{"file": s["file"], "section": s["section"]} for s in tr.selected],
            "seconds": tr.seconds, "tokens": tr.tokens,
        }
        if q["answerable"]:
            res["recall_at_3"] = recall_at(tr.retrieved, q["expected_files"], RERANK_TOP)
            res["recall_at_10"] = recall_at(tr.retrieved, q["expected_files"], 10)  # בראש 10 בסדר הדירוג הסופי
            res["recall_pool"] = recall_at(tr.retrieved, q["expected_files"], max(len(tr.retrieved), 1))  # בכל המועמדים שנשלפו
            res["rr"] = 1 / ranks[0] if ranks else 0.0
            res["final_recall"] = len({s["file"] for s in tr.selected} & set(q["expected_files"])) / len(q["expected_files"])
        # Faithfulness: האם התשובה נתמכת בקטעים שנשלחו למודל. סירוב נחשב נאמן מעצם הגדרתו.
        if tr.is_refusal or not tr.selected:
            res["faithfulness"] = 1.0
        else:
            ctx = "\n\n".join(f"[קטע {i}]\n{s['text']}" for i, s in enumerate(tr.selected, 1))
            d, a, b = judge(judge_llm, FAITH_SYSTEM, f"קטעים:\n{ctx}\n\nתשובה: {tr.answer}")
            judge_in, judge_out = judge_in + a, judge_out + b
            res["faithfulness"] = float(d.get("faithful", 0))
        # Answer correctness. בשאלה ללא תשובה נכון לסרב; בשאלה עם תשובה סירוב הוא שגיאה.
        if not q["answerable"]:
            res["correctness"] = 1.0 if tr.is_refusal else 0.0
        elif tr.is_refusal:
            res["correctness"] = 0.0
        else:
            d, a, b = judge(judge_llm, CORRECT_SYSTEM,
                            f"שאלה: {q['question']}\nתשובה נכונה: {q['expected_answer']}\nתשובת המערכת: {tr.answer}")
            judge_in, judge_out = judge_in + a, judge_out + b
            res["correctness"] = float(d.get("correct", 0))
        results.append(res)
        print(f"  שאלה {q['id']:>2} | סירוב={tr.is_refusal!s:5} | נכונות={res['correctness']} | נאמנות={res['faithfulness']} | {tr.seconds}s")

    ans = [r for r in results if r["answerable"]]
    unans = [r for r in results if not r["answerable"]]
    pipe_in = sum(r["tokens"]["rerank_prompt"] + r["tokens"]["answer_prompt"] for r in results)
    pipe_out = sum(r["tokens"]["rerank_completion"] + r["tokens"]["answer_completion"] for r in results)
    p = PRICES[LLM_MODEL]
    summary = {
        "n_questions": len(results), "n_answerable": len(ans), "n_unanswerable": len(unans),
        "recall_at_3": mean(r["recall_at_3"] for r in ans),
        "recall_at_10": mean(r["recall_at_10"] for r in ans),
        "recall_pool": mean(r["recall_pool"] for r in ans),
        "avg_chunks_approved": mean(sum(1 for t in r["retrieved"] if t["rerank_score"] >= RERANK_MIN_RELEVANT) for r in results),
        "avg_chunks_sent": mean(len(r["selected"]) for r in results),
        "mrr": mean(r["rr"] for r in ans),
        "final_context_recall": mean(r["final_recall"] for r in ans),
        "faithfulness": mean(r["faithfulness"] for r in results if not r["is_refusal"]),
        "answer_correctness_answerable": mean(r["correctness"] for r in ans),
        "correct_refusal_rate": mean(r["correctness"] for r in unans),
        "false_refusal_rate": mean(1.0 if r["is_refusal"] else 0.0 for r in ans),
        "game_filters": sum(1 for r in results if r["games"]),
        "retries": sum(1 for r in results if r["retried"]),
        "avg_seconds": mean(r["seconds"] for r in results),
        "total_wall_seconds": round(time.time() - t_start, 1),
        "pipeline_llm_tokens": pipe_in + pipe_out,
        "avg_pipeline_llm_tokens_per_question": (pipe_in + pipe_out) / len(results),
        "judge_tokens": judge_in + judge_out,
        "actual_llm_cost_usd": (pipe_in + judge_in) * p["input"] / 1e6 + (pipe_out + judge_out) * p["output"] / 1e6,
    }
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    label = f"_{args.label}" if args.label else ""
    config = {
        "llm": LLM_MODEL, "embedding": args.embed_model, "chunking": CHUNKING_METHOD,
        "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP, "retrieve_k": RETRIEVE_K,
        "rerank_top": RERANK_TOP, "rerank_group": RERANK_GROUP,
        "rerank_min_relevant": RERANK_MIN_RELEVANT, "send_only_relevant": SEND_ONLY_RELEVANT, "answer_mode": ANSWER_MODE,
    }
    report = {"timestamp": datetime.now().isoformat(timespec="seconds"), "label": args.label,
              "config": config, "summary": summary, "results": results}
    REPORTS.mkdir(exist_ok=True)
    base = REPORTS / f"eval_{stamp}{label}"
    base.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    base.with_suffix(".md").write_text(render_md(report), encoding="utf-8")
    print(f"\nהדוח נשמר: {base.with_suffix('.md').name}")
    for k, v in summary.items():
        print(f"  {k}: {round(v, 3) if isinstance(v, float) else v}")
    return 0


def fmt(v, pct=False):
    if v is None:
        return "אין נתונים"
    return f"{v * 100:.0f}%" if pct else f"{v:.2f}"


def render_md(r: dict) -> str:
    s, c = r["summary"], r["config"]
    lines = [
        f"# דוח הערכה {r['timestamp']} {r['label']}".strip(), "",
        "הסט כולל 10 שאלות בלבד, ולכן המספרים הם אינדיקציה ראשונית ולא מדידה סטטיסטית.", "",
        "## הגדרות", "",
        f"- מודל שפה: `{c['llm']}`, מודל Embedding: `{c['embedding']}`",
        f"- חלוקה: {c['chunking']}, {c['chunk_size']} טוקנים, חפיפה {c['chunk_overlap']}",
        f"- אחזור: {c['retrieve_k']} קטעים, למודל נשלחים {c['rerank_top']}",
        f"- סף רלוונטיות לדירוג: {c['rerank_min_relevant']}, שליחת קטעים רלוונטיים בלבד: {c.get('send_only_relevant', 'לא ידוע')}, מצב תשובה: {c.get('answer_mode', 'plain')}", "",
        "## מדדים", "",
        "| מדד | ערך |", "|---|---|",
        f"| Recall@3 (שאלות עם תשובה) | {fmt(s['recall_at_3'], True)} |",
        f"| Recall@10 (בראש 10 אחרי דירוג) | {fmt(s['recall_at_10'], True)} |",
        f"| Recall בכל המועמדים שנשלפו (K={c['retrieve_k']}) | {fmt(s['recall_pool'], True)} |",
        f"| קטעים שהמדרג אישר / נשלחו למודל (ממוצע לשאלה) | {s['avg_chunks_approved']:.2f} / {s['avg_chunks_sent']:.2f} |",
        f"| MRR | {fmt(s['mrr'])} |",
        f"| כיסוי בקטעים שנשלחו למודל | {fmt(s['final_context_recall'], True)} |",
        f"| Faithfulness (תשובות שאינן סירוב) | {fmt(s['faithfulness'], True)} |",
        f"| Answer correctness (שאלות עם תשובה) | {fmt(s['answer_correctness_answerable'], True)} |",
        f"| אחוז סירובים נכונים (שאלות ללא תשובה) | {fmt(s['correct_refusal_rate'], True)} |",
        f"| אחוז סירובים שגויים (שאלות עם תשובה) | {fmt(s['false_refusal_rate'], True)} |",
        f"| שאלות עם סינון לפי שם משחק | {s['game_filters']} |",
        f"| חיפושים חוזרים | {s['retries']} |",
        f"| זמן ממוצע לשאלה (שניות) | {s['avg_seconds']:.1f} |",
        f"| זמן כולל להרצה, כולל שופט (שניות) | {s['total_wall_seconds']} |",
        f"| טוקנים ב-LLM בתהליך המענה (סך הכל / ממוצע לשאלה) | {s['pipeline_llm_tokens']} / {s['avg_pipeline_llm_tokens_per_question']:.0f} |",
        f"| טוקנים של השופט (Faithfulness ו-Correctness) | {s['judge_tokens']} |",
        f"| עלות LLM בפועל (לפי המחירון בהגדרות) | ${s['actual_llm_cost_usd']:.4f} |", "",
        "## תוצאות לפי שאלה", "",
        "| # | קטגוריה | סירוב | נכונות | נאמנות | Recall@3 | RR | משחק בשאלה | חיפוש חוזר | שניות |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for x in r["results"]:
        lines.append(
            f"| {x['id']} | {x['category']} | {'כן' if x['is_refusal'] else 'לא'} | {x['correctness']} | {x['faithfulness']} | "
            f"{fmt(x.get('recall_at_3'), True) if 'recall_at_3' in x else '-'} | {fmt(x.get('rr')) if 'rr' in x else '-'} | "
            f"{', '.join(x['games']) or 'ללא'} | {'כן' if x['retried'] else 'לא'} | {x['seconds']} |"
        )
    lines += ["", "## תשובות", ""]
    for x in r["results"]:
        lines += [f"**{x['id']}. {x['question']}**", f"- תשובה נכונה: {x['expected_answer']}", f"- תשובת המערכת: {x['answer']}", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
