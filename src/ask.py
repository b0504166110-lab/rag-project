"""שאלה משורת הפקודה, להצגת התשובה והפרטים הטכניים (לבדיקה בלבד). השאלות נרשמות ביומן.
  python src/ask.py "כמה זמן אחריות יש?"
  python src/ask.py --source examples "שאלה 1" "שאלה 2"   (שאלות שישמשו את docs/examples.md)
"""
import argparse
import json
import sys
from dataclasses import asdict

from qlog import log_trace
from rag import build_pipeline


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("questions", nargs="+")
    ap.add_argument("--source", default="cli", help="תווית ביומן (ברירת מחדל: cli)")
    args = ap.parse_args()
    pipe = build_pipeline()
    for q in args.questions:
        tr = pipe.ask(q)
        log_trace(tr, source=args.source, embed_model=pipe.embed.model_name)
        print("\nשאלה:", q)
        print("תשובה:", tr.answer)
        d = asdict(tr)
        for k in ("games", "retried", "rerank_failed", "seconds", "tokens"):
            print(f"  {k}: {json.dumps(d[k], ensure_ascii=False)}")
        print("  נבחרו:", json.dumps([{"file": s["file"], "section": s["section"]} for s in d["selected"]], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
