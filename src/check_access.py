"""בדיקת גישה קצרה לשני המודלים. לא מדפיס את המפתח."""
import sys

from openai import OpenAI

from config import EMBED_MODEL, LLM_MODEL, get_api_key


def main() -> int:
    key = get_api_key()
    if not key:
        print("חסר OPENAI_API_KEY בקובץ .env")
        return 1
    client = OpenAI(api_key=key)
    ok = True
    try:
        r = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "ענה במילה אחת: שלום"}],
            max_tokens=10,
        )
        print(f"[מודל שפה] {LLM_MODEL}: תקין, טוקנים={r.usage.total_tokens}")
    except Exception as e:
        ok = False
        print(f"[מודל שפה] {LLM_MODEL}: נכשל ({type(e).__name__})")
    try:
        r = client.embeddings.create(model=EMBED_MODEL, input="בדיקה")
        print(f"[Embedding] {EMBED_MODEL}: תקין, ממד={len(r.data[0].embedding)}")
    except Exception as e:
        ok = False
        print(f"[Embedding] {EMBED_MODEL}: נכשל ({type(e).__name__})")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
