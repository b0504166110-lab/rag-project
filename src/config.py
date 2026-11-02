"""כל ההגדרות של המערכת במקום אחד."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# מודלים
LLM_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-large"

# נתיבים
DOCS_DIR = ROOT / "data"
CHROMA_DIR = ROOT / "chroma_db"
LOGS_DIR = ROOT / "logs"

# חלוקה לקטעים: השיטה נבחרת כאן, ומימושים נוספים נרשמים ב-chunking.py
# sentence: לפי גודל ומשפטים (ברירת המחדל). markdown: לפי מבנה המסמך (בונוס). אפשר לדרוס בניסוי עם RAG_CHUNKING_METHOD.
CHUNKING_METHOD = os.getenv("RAG_CHUNKING_METHOD", "sentence")
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


# מענה על שאלות
# כמה קטעים נשלפים לפני הדירוג (המפרט: 10). 20 בקריאת דירוג אחת נוסה ובוטל (בעיה 11), ולכן הדירוג נעשה
# בקבוצות של RERANK_GROUP קטעים, קריאה נפרדת לכל קבוצה. אפשר לדרוס בניסוי (RAG_RETRIEVE_K=10 מחזיר לקריאה אחת).
RETRIEVE_K = int(os.getenv("RAG_RETRIEVE_K", "20"))
RERANK_GROUP = int(os.getenv("RAG_RERANK_GROUP", "10"))
RERANK_TOP = 3  # כמה קטעים נשלחים למודל השפה אחרי הדירוג
# לשלוח למודל התשובה רק קטעים שהמדרג לא פסל (ציון לפחות RERANK_MIN_RELEVANT), ולפחות את הקטע הראשון.
# אפשר לדרוס בניסוי עם RAG_SEND_ONLY_RELEVANT=0.
SEND_ONLY_RELEVANT = os.getenv("RAG_SEND_ONLY_RELEVANT", "1") == "1"
# קטע נחשב מתאים אם ציון הדירוג שלו (0 עד 10) הוא לפחות כך.
# אם אחרי סינון אין אף קטע מתאים, מחפשים שוב בלי סינון.
RERANK_MIN_RELEVANT = 5
GENERAL_SCOPE = "כללי"  # ערך game_scope למסמכים שאינם שייכים למשחק מסוים
# plain: תשובה ישירה. evidence: המודל מצטט קודם את המשפט שעונה על השאלה ואז עונה (סירוב רק אם אין ציטוט).
# אפשר לדרוס בניסוי עם RAG_ANSWER_MODE.
ANSWER_MODE = os.getenv("RAG_ANSWER_MODE", "evidence")
NO_ANSWER = "אין לי מספיק מידע במאגר כדי לענות על השאלה הזו."



# מחירון להערכת עלות בלבד (דולר למיליון טוקנים). יש לאמת מול המחירון הנוכחי של OpenAI.
PRICES = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "text-embedding-3-large": {"input": 0.13},
    "text-embedding-3-small": {"input": 0.02},
}


def collection_name(embed_model: str | None = None, method: str | None = None) -> str:
    """אוסף נפרד ב-Chroma לכל מודל Embedding ולכל שיטת חלוקה (שיטת ברירת המחדל שומרת על השם המקורי)."""
    base = f"docs_{embed_model or EMBED_MODEL}"
    method = method or CHUNKING_METHOD
    return base if method == "sentence" else f"{base}__{method}"


def get_api_key() -> str | None:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    return key or None
