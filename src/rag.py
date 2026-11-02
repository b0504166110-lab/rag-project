"""תהליך המענה על שאלה: התאמת משחק, אחזור, חיפוש חוזר, דירוג ותשובה."""
import json
import re
import time
from dataclasses import dataclass, field

import tiktoken
from llama_index.core.llms import ChatMessage
from llama_index.core.schema import MetadataMode
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters

from config import (
    DOCS_DIR,
    GENERAL_SCOPE,
    ANSWER_MODE,
    NO_ANSWER,
    RERANK_GROUP,
    RERANK_TOP,
    RERANK_MIN_RELEVANT,
    RETRIEVE_K,
    SEND_ONLY_RELEVANT,
)
from loader import game_names

ENC = tiktoken.get_encoding("cl100k_base")

RERANK_SYSTEM = (
    "אתה מדרג עד כמה כל קטע טקסט עוזר לענות על שאלה. "
    "החזר JSON בלבד בפורמט {\"scores\": [..]} עם ציון שלם מ-0 עד 10 לכל קטע, לפי הסדר שבו הוצגו. "
    "10 = עונה ישירות על השאלה, 0 = לא קשור. אל תוסיף טקסט אחר."
)

ANSWER_SYSTEM = (
    "אתה עוזר שירות של חברת משחקי קופסה. ענה בעברית, בקצרה ובבהירות, "
    "אך ורק על סמך הקטעים שמופיעים להלן. אל תוסיף ידע חיצוני ואל תנחש.\n"
    f"אם הקטעים אינם כוללים מידע מספיק כדי לענות, ענה בדיוק במשפט הזה ותו לא: {NO_ANSWER}\n"
    "לכל קטע מצורפים סוג מסמך, גרסה ותאריך עדכון. "
    "אם יש גרסאות שונות של אותה מדיניות או של אותו קטלוג והשאלה לא מציינת גרסה, "
    "ענה לפי הגרסה העדכנית ביותר, ואם רלוונטי ציין בקצרה מה היה בגרסה הישנה.\n"
    "כשהשאלה משווה בין גרסאות או מהדורות, ציין במפורש איזה ערך שייך לכל גרסה, והעתק מספרים בדיוק כפי שהם בקטעים."
)

EVIDENCE_SYSTEM = (
    ANSWER_SYSTEM
    + "\nהחזר JSON בלבד בפורמט {\"evidence\": \"...\", \"answer\": \"...\"}. "
    "בשדה evidence העתק מילה במילה את המשפט או המשפטים מהקטעים שעונים על השאלה, ואם אין כאלה השאר מחרוזת ריקה. "
    "בשדה answer כתוב את התשובה על סמך ה-evidence בלבד, במשפט שלם שחוזר על נושא השאלה "
    f"(למשל 'ים של אבנים עולה 149 ₪' ולא '149 ₪'), ואם evidence ריק כתוב בדיוק: {NO_ANSWER}"
)


@dataclass
class Trace:
    """כל מה שקרה בטיפול בשאלה אחת, לצורך היומן והערכה. אינו מוצג למשתמש."""
    question: str
    answer: str = ""
    is_refusal: bool = False
    games: list = field(default_factory=list)
    filters: dict = field(default_factory=dict)
    retried: bool = False
    retrieved: list = field(default_factory=list)  # file, section, doc_type, vector_score, rerank_score
    selected: list = field(default_factory=list)
    rerank_failed: bool = False
    seconds: float = 0.0
    tokens: dict = field(default_factory=dict)


def _norm(s: str) -> str:
    return re.sub(r"[\"'׳״\s]+", " ", s).strip()


def _usage(resp) -> tuple[int, int]:
    u = getattr(getattr(resp, "raw", None), "usage", None)
    return (getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0)) if u else (0, 0)


class RagPipeline:
    def __init__(self, index, embed_model, llm, rerank_llm):
        self.index, self.embed, self.llm, self.rerank_llm = index, embed_model, llm, rerank_llm
        self.games = game_names(DOCS_DIR)

    # שם משחק מהמאגר שמופיע בשאלה (אין סיווג אוטומטי לפי סוג מסמך; הוסר, ראו docs/problems.md)
    def match_games(self, question: str) -> list[str]:
        q = _norm(question)
        return [g for g in self.games if _norm(g) in q]

    # שלב 3: אחזור
    def retrieve(self, question: str, games: list[str]):
        flt = []
        if games:
            flt.append(MetadataFilter(key="game_scope", value=games + [GENERAL_SCOPE], operator=FilterOperator.IN))
        retriever = self.index.as_retriever(
            similarity_top_k=RETRIEVE_K, filters=MetadataFilters(filters=flt) if flt else None
        )
        return retriever.retrieve(question)

    # שלב 5: דירוג מחדש עם מודל השפה
    # המדרג יציב עד 10 קטעים בקריאה אחת (בעיה 11), ולכן מדרגים בקבוצות ומחברים את הציונים לפי סדר האחזור
    def rerank(self, question: str, nodes) -> tuple[list[float], tuple[int, int], bool]:
        scores: list[float] = []
        used = (0, 0)
        failed = False
        for start in range(0, len(nodes), RERANK_GROUP):
            sc, (a, b), f = self._rerank_group(question, nodes[start:start + RERANK_GROUP])
            scores += sc
            used = (used[0] + a, used[1] + b)
            failed = failed or f
        if failed:
            # אם הפלט של קבוצה כלשהי אינו תקין: שומרים על סדר האחזור בכל הרשימה
            scores = [float(len(nodes) - i) for i in range(len(nodes))]
        return scores, used, failed

    def _rerank_group(self, question: str, nodes) -> tuple[list[float], tuple[int, int], bool]:
        parts = []
        for i, n in enumerate(nodes, 1):
            parts.append(f"[קטע {i}]\n{n.node.get_content(metadata_mode=MetadataMode.LLM)}")
        msg = [
            ChatMessage(role="system", content=RERANK_SYSTEM),
            ChatMessage(role="user", content=f"שאלה: {question}\n\n" + "\n\n".join(parts)),
        ]
        resp = self.rerank_llm.chat(msg)
        usage = _usage(resp)
        try:
            scores = [float(s) for s in json.loads(resp.message.content)["scores"]]
            if len(scores) != len(nodes):
                raise ValueError("מספר ציונים שגוי")
            return scores, usage, False
        except Exception:
            # אם הפלט אינו תקין: שומרים על סדר האחזור
            return [float(len(nodes) - i) for i in range(len(nodes))], usage, True

    # שלב 6: תשובה
    def answer(self, question: str, nodes) -> tuple[str, tuple[int, int]]:
        ctx = "\n\n".join(
            f"[קטע {i}]\n{n.node.get_content(metadata_mode=MetadataMode.LLM)}" for i, n in enumerate(nodes, 1)
        )
        msg = [
            ChatMessage(role="system", content=ANSWER_SYSTEM),
            ChatMessage(role="user", content=f"קטעים:\n{ctx}\n\nשאלה: {question}"),
        ]
        if ANSWER_MODE == "evidence":
            msg[0] = ChatMessage(role="system", content=EVIDENCE_SYSTEM)
            resp = self.rerank_llm.chat(msg)  # מודל במצב JSON
            try:
                data = json.loads(resp.message.content)
                ans = str(data.get("answer", "")).strip() if str(data.get("evidence", "")).strip() else NO_ANSWER
            except Exception:
                ans = resp.message.content.strip()
            return ans or NO_ANSWER, _usage(resp)
        resp = self.llm.chat(msg)
        return resp.message.content.strip(), _usage(resp)

    def ask(self, question: str) -> Trace:
        t0 = time.time()
        tr = Trace(question=question)
        tr.games = self.match_games(question)
        tr.filters = {"games": tr.games}

        nodes = self.retrieve(question, tr.games)
        filtered = bool(tr.games)
        rr_in = rr_out = ans_in = ans_out = 0
        scores: list[float] = []
        # שלב 4: חיפוש חוזר בלי סינון אם אחרי הסינון לא נמצא דבר מתאים
        if filtered and not nodes:
            tr.retried = True
            nodes = self.retrieve(question, [])
        if nodes:
            scores, (rr_in, rr_out), tr.rerank_failed = self.rerank(question, nodes)
            if filtered and not tr.retried and max(scores) < RERANK_MIN_RELEVANT:
                tr.retried = True
                nodes = self.retrieve(question, [])
                scores, (a, b), tr.rerank_failed = self.rerank(question, nodes)
                rr_in, rr_out = rr_in + a, rr_out + b
        if nodes:
            # בשוויון בציון הדירוג מעדיפים מסמך עדכני יותר, ואחר כך ציון אחזור גבוה
            order = sorted(
                range(len(nodes)),
                key=lambda i: (scores[i], nodes[i].node.metadata["updated"], nodes[i].score),
                reverse=True,
            )
            tr.retrieved = [
                {
                    "file": n.node.metadata["file_name"],
                    "section": n.node.metadata["section"],
                    "doc_type": n.node.metadata["doc_type"],
                    "vector_score": round(float(n.score), 4),
                    "rerank_score": scores[i],
                }
                for i, n in ((i, nodes[i]) for i in order)  # בסדר הדירוג הסופי
            ]
            keep = order[:RERANK_TOP]
            if SEND_ONLY_RELEVANT and not tr.rerank_failed:
                # קטעים שהמדרג פסל מסיחים את המודל וגורמים לסירוב שגוי; הקטע הראשון נשלח תמיד
                keep = [i for i in keep if scores[i] >= RERANK_MIN_RELEVANT] or keep[:1]
            top = [nodes[i] for i in keep]
            tr.selected = [
                {
                    "file": n.node.metadata["file_name"],
                    "section": n.node.metadata["section"],
                    "text": n.node.get_content(metadata_mode=MetadataMode.LLM),  # כפי שהמודל ראה, כולל מטא-דאטה
                }
                for n in top
            ]
            tr.answer, (ans_in, ans_out) = self.answer(question, top)
        else:
            tr.answer = NO_ANSWER
        tr.is_refusal = tr.answer.startswith(NO_ANSWER[:20])
        tr.seconds = round(time.time() - t0, 2)
        tr.tokens = {
            "rerank_prompt": rr_in, "rerank_completion": rr_out,
            "answer_prompt": ans_in, "answer_completion": ans_out,
            "total_llm": rr_in + rr_out + ans_in + ans_out,
            "question_embedding_est": len(ENC.encode(question)),
        }
        return tr


def build_pipeline(rebuild: bool = False, embed_model_name: str | None = None):
    from indexer import get_index
    from models import get_embed_model, get_llm

    embed = get_embed_model(embed_model_name)
    index, _ = get_index(embed, rebuild=rebuild)
    return RagPipeline(index, embed, get_llm(), get_llm(json_mode=True))
