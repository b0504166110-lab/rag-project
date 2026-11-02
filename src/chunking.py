"""שיטות חלוקה לקטעים. כדי להוסיף שיטה: להגדיר פונקציה ולרשום אותה ב-METHODS.

כל שיטה מחזירה אובייקט עם get_nodes_from_documents([doc]).
"""
import re

from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter

from config import CHUNKING_METHOD, CHUNK_OVERLAP, CHUNK_SIZE


def _sentence():
    return SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


class MarkdownThenSentence:
    """חלוקה לפי מבנה המסמך (כותרת לכל קטע). קטע ארוך מדי מחולק גם לפי גודל.
    קובצי PDF אינם מחולקים לפי מבנה: הם נשארים בחלוקה לפי גודל ומשפטים."""

    def __init__(self):
        self.md = MarkdownNodeParser()
        self.sentence = _sentence()

    def get_nodes_from_documents(self, docs):
        out = []
        for doc in docs:
            if doc.metadata.get("file_name", "").lower().endswith(".pdf"):
                out += self.sentence.get_nodes_from_documents([doc])
                continue
            for section in self.md.get_nodes_from_documents([doc]):
                first_line = section.text.splitlines()[0] if section.text else ""
                m = re.match(r"^#+\s+(.*)", first_line)
                title = m.group(1).strip() if m else "פתיחה"
                part = Document(text=section.text, metadata=dict(doc.metadata), id_=doc.id_)
                part.excluded_embed_metadata_keys = list(doc.excluded_embed_metadata_keys)
                part.excluded_llm_metadata_keys = list(doc.excluded_llm_metadata_keys)
                for node in self.sentence.get_nodes_from_documents([part]):
                    node.metadata["section"] = title
                    out.append(node)
        return out


METHODS = {"sentence": _sentence, "markdown": MarkdownThenSentence}


def get_splitter(method: str | None = None):
    name = method or CHUNKING_METHOD
    if name not in METHODS:
        raise ValueError(f"שיטת חלוקה לא מוכרת: {name}. זמינות: {', '.join(METHODS)}")
    return METHODS[name]()
