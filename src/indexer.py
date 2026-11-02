"""בניית אינדקס ב-ChromaDB, וטעינתו מהדיסק בלי Embedding מחדש."""
import bisect
import hashlib

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import MetadataMode
from llama_index.vector_stores.chroma import ChromaVectorStore

from chunking import get_splitter
from config import CHROMA_DIR, CHUNK_OVERLAP, CHUNK_SIZE, CHUNKING_METHOD, DOCS_DIR, collection_name
from loader import load_documents, section_index


MAX_SECTIONS_IN_META = 3


def sections_of(heads, start: int, end: int) -> str:
    """כותרות כל הסעיפים שהקטע נוגע בהם: זה שמתחיל לפניו (אם יש) וגם אלה שבתוכו."""
    starts = [h[0] for h in heads]
    first = max(bisect.bisect_right(starts, start) - 1, 0)
    last = max(bisect.bisect_left(starts, end) - 1, first)
    titles = [h[1] for h in heads[first:last + 1]]
    if not titles:
        return "פתיחה"
    shown = titles[:MAX_SECTIONS_IN_META]
    return " | ".join(shown) + (" ..." if len(titles) > len(shown) else "")


def build_nodes(docs):
    """חלוקה לקטעים ושיוך כל קטע לכותרות הסעיפים שבהם הוא נמצא."""
    splitter = get_splitter()
    nodes = []
    for doc in docs:
        heads = section_index(doc.text)
        for node in splitter.get_nodes_from_documents([doc]):
            if "section" not in node.metadata:  # שיטות חלוקה לפי מבנה מציבות את הסעיף בעצמן
                start = node.start_char_idx or 0
                end = node.end_char_idx if node.end_char_idx is not None else start + len(node.text)
                node.metadata["section"] = sections_of(heads, start, end)
            nodes.append(node)
    return nodes


def docs_fingerprint(docs) -> str:
    h = hashlib.sha256()
    for d in docs:
        h.update(d.id_.encode() + d.text.encode() + repr(sorted(d.metadata.items())).encode())
    h.update(f"{CHUNKING_METHOD}|{CHUNK_SIZE}|{CHUNK_OVERLAP}".encode())
    return h.hexdigest()[:16]


def _collection(client, name: str):
    return client.get_or_create_collection(name, configuration={"hnsw": {"space": "cosine"}})


def get_index(embed_model, rebuild: bool = False, collection: str | None = None):
    """טוען אינדקס קיים; בונה (עם Embedding) רק אם האוסף ריק או ביקשו rebuild."""
    name = collection or collection_name(embed_model.model_name)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if rebuild:
        try:
            client.delete_collection(name)
        except Exception:
            pass
    col = _collection(client, name)
    store = ChromaVectorStore(chroma_collection=col)
    if col.count() > 0:
        return VectorStoreIndex.from_vector_store(store, embed_model=embed_model), False
    docs = load_documents(DOCS_DIR)
    nodes = build_nodes(docs)
    index = VectorStoreIndex(
        nodes,
        storage_context=StorageContext.from_defaults(vector_store=store),
        embed_model=embed_model,
        show_progress=True,
    )
    col.modify(metadata={"fingerprint": docs_fingerprint(docs)})
    return index, True


def stale_warning(collection: str | None = None, embed_model_name: str | None = None) -> str | None:
    """אזהרה אם המסמכים או הגדרות החלוקה השתנו מאז בניית האינדקס."""
    name = collection or collection_name(embed_model_name)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        col = client.get_collection(name)
    except Exception:
        return None
    if (col.metadata or {}).get("fingerprint") != docs_fingerprint(load_documents(DOCS_DIR)):
        return "המסמכים או הגדרות החלוקה השתנו מאז בניית האינדקס. להרצה: python src/ingest.py --rebuild"
    return None


def embed_text_of(node) -> str:
    return node.get_content(metadata_mode=MetadataMode.EMBED)
