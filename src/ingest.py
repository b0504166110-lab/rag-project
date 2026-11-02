"""טעינה, חלוקה ואינדוקס. שימוש:
  python src/ingest.py --dry-run   חלוקה והערכת עלות, בלי קריאות API
  python src/ingest.py             בניית אינדקס (אם אינו קיים)
  python src/ingest.py --rebuild   מחיקה ובנייה מחדש
"""
import argparse
import collections
import sys

import tiktoken

from config import CHROMA_DIR, DOCS_DIR, EMBED_MODEL
from indexer import build_nodes, embed_text_of, get_index, stale_warning
from loader import load_documents
from models import MissingApiKey, get_embed_model

PRICE_PER_M = {"text-embedding-3-large": 0.13, "text-embedding-3-small": 0.02}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--model", default=EMBED_MODEL)
    args = ap.parse_args()

    docs = load_documents(DOCS_DIR)
    nodes = build_nodes(docs)
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = sum(len(enc.encode(embed_text_of(n))) for n in nodes)
    cost = tokens / 1e6 * PRICE_PER_M.get(args.model, 0.13)
    per_type = collections.Counter(d.metadata["doc_type"] for d in docs)
    print(f"מסמכים: {len(docs)} | קטעים: {len(nodes)} | טוקנים ל-Embedding: {tokens} | עלות משוערת (רק אם יבוצע Embedding): ${cost:.4f}")
    for t, c in per_type.items():
        print(f"  {t}: {c} מסמכים")
    sizes = [len(enc.encode(n.get_content())) for n in nodes]
    print(f"גודל קטע בטוקנים: מינימום {min(sizes)}, ממוצע {sum(sizes) // len(sizes)}, מקסימום {max(sizes)}")
    if args.dry_run:
        return 0

    try:
        embed = get_embed_model(args.model)
    except MissingApiKey as e:
        print(e)
        return 1
    index, built = get_index(embed, rebuild=args.rebuild)
    print("האינדקס נבנה (בוצע Embedding)." if built else f"האינדקס כבר קיים ב-{CHROMA_DIR.name}, לא בוצע Embedding מחדש.")
    warn = stale_warning(embed_model_name=args.model)
    if warn:
        print("אזהרה:", warn)
    return 0


if __name__ == "__main__":
    sys.exit(main())
