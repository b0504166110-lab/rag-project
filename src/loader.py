"""טעינת מסמכי Markdown ו-PDF למסמכי LlamaIndex, כולל מטא-דאטה."""
import re
from pathlib import Path

from bidi import get_display
from llama_index.core import Document
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTTextLine

from config import GENERAL_SCOPE

HEADER_KEYS = {
    "סוג מסמך": "doc_type",
    "משחק": "game",
    "תאריך עדכון": "updated",
    "גרסה": "version",
}
HEADER_RE = re.compile(r"^(" + "|".join(HEADER_KEYS) + r"):\s*(.+?)\s*$")
HEADING_RE = re.compile(r"^## .+$", re.M)
PDF_HEADING_RE = re.compile(r"^\d+\.\s")  # כותרת סעיף ב-PDF, למשל "2. תכולת המשחק"
PDF_ITEM_RE = re.compile(r"^\d+\.\d+\.")  # פריט, למשל "2.1."
CATALOG_ROW_RE = re.compile(r"^\|\s*SH-\d+\s*\|\s*([^|]+?)\s*\|", re.M)
LINE_TOLERANCE = 3.0  # נקודות; קטעי טקסט באותו גובה נחשבים לאותה שורה


def _read_pdf_lines(path: Path) -> list[str]:
    """חילוץ שורות לפי מיקום בעמוד, ואז המרה מסדר ויזואלי לסדר לוגי."""
    lines: list[str] = []
    for page in extract_pages(str(path)):
        boxes = []
        for el in page:
            if isinstance(el, LTTextContainer):
                for ln in el:
                    if isinstance(ln, LTTextLine) and ln.get_text().strip():
                        boxes.append((ln.y1, ln.x1, ln.get_text().strip()))
        boxes.sort(key=lambda b: -b[0])
        rows: list[list[tuple]] = []
        for b in boxes:
            if rows and abs(rows[-1][0][0] - b[0]) <= LINE_TOLERANCE:
                rows[-1].append(b)
            else:
                rows.append([b])
        for row in rows:
            row.sort(key=lambda b: -b[1])  # מימין לשמאל
            lines.append(" ".join(get_display(b[2], base_dir="R") for b in row))
    return lines


def _pdf_to_markdown(lines: list[str]) -> str:
    """איחוד שורות שנשברו, וסימון כותרות כמו ב-Markdown."""
    out: list[str] = []
    for i, ln in enumerate(lines):
        if i == 0:
            out.append("# " + ln)
        elif PDF_ITEM_RE.match(ln) or HEADER_RE.match(ln):
            out.append(ln)
        elif PDF_HEADING_RE.match(ln):
            out.append("## " + ln)
        elif out:
            out[-1] += " " + ln  # המשך של שורה שנשברה
        else:
            out.append(ln)
    return "\n".join(out)


def read_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return _pdf_to_markdown(_read_pdf_lines(path))
    if path.suffix.lower() == ".md":
        return path.read_text(encoding="utf-8")
    raise ValueError(f"סוג קובץ לא נתמך: {path.name}")


def parse_header(text: str) -> tuple[dict, str]:
    """מוציא את שורות הכותרת (סוג מסמך, משחק, תאריך, גרסה) למטא-דאטה."""
    meta, body = {}, []
    for ln in text.splitlines():
        m = HEADER_RE.match(ln.strip())
        if m and len(meta) < len(HEADER_KEYS):
            meta[HEADER_KEYS[m.group(1)]] = m.group(2)
        else:
            body.append(ln)
    return meta, re.sub(r"\n{3,}", "\n\n", "\n".join(body)).strip()


def section_index(text: str) -> list[tuple[int, str]]:
    """מיקומי הכותרות בטקסט, כדי לשייך כל קטע לסעיף שלו."""
    return [(m.start(), m.group(0)[3:].strip()) for m in HEADING_RE.finditer(text)]


def load_documents(docs_dir: Path) -> list[Document]:
    docs = []
    for path in sorted(docs_dir.iterdir()):
        if path.suffix.lower() not in (".md", ".pdf"):
            continue
        meta, body = parse_header(read_text(path))
        missing = {"doc_type", "updated", "version"} - meta.keys()
        if missing:
            raise ValueError(f"{path.name}: חסרים שדות כותרת: {missing}")
        meta["file_name"] = path.name
        meta["game_scope"] = meta.get("game") or GENERAL_SCOPE  # לסינון לפי משחק
        doc = Document(text=body, metadata=meta, id_=path.name)
        doc.excluded_embed_metadata_keys = ["file_name", "game_scope"]
        doc.excluded_llm_metadata_keys = ["file_name", "game_scope"]
        docs.append(doc)
    return docs


def game_names(docs_dir: Path) -> list[str]:
    """שמות המשחקים במאגר: ממטא-דאטה של המסמכים ומטבלאות הקטלוג."""
    names = {d.metadata["game"] for d in load_documents(docs_dir) if d.metadata.get("game")}
    for path in docs_dir.glob("catalog_*.md"):
        for m in CATALOG_ROW_RE.finditer(path.read_text(encoding="utf-8")):
            names.add(re.sub(r"\s*\(.*\)$", "", m.group(1)).strip())
    return sorted(names, key=len, reverse=True)
