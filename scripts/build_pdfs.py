"""הופך את קובצי ה-Markdown שב-pdf_src לקובצי PDF בעברית (RTL) בתיקיית data.

משתמש ב-Edge במצב headless להדפסה ל-PDF, כך שהטקסט נשאר טקסט לוגי וניתן לחילוץ.
"""
import html
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "pdf_src"
OUT = ROOT / "data"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")

CSS = """
body { font-family: Arial, 'Segoe UI', sans-serif; direction: rtl; text-align: right; font-size: 13pt; line-height: 1.6; }
h1 { font-size: 22pt; } h2 { font-size: 16pt; margin-top: 18pt; }
table { border-collapse: collapse; width: 100%; direction: rtl; }
td, th { border: 1px solid #888; padding: 4px 8px; text-align: right; }
"""


def md_to_html(text: str) -> str:
    out, table = [], []

    def flush_table():
        if not table:
            return
        rows = [r for r in table if not re.match(r"^\|[\s\-|]+\|$", r)]
        out.append("<table>")
        for i, r in enumerate(rows):
            cells = [c.strip() for c in r.strip("|").split("|")]
            tag = "th" if i == 0 else "td"
            out.append("<tr>" + "".join(f"<{tag}>{html.escape(c)}</{tag}>" for c in cells) + "</tr>")
        out.append("</table>")
        table.clear()

    for line in text.splitlines():
        if line.startswith("|"):
            table.append(line)
            continue
        flush_table()
        if line.startswith("## "):
            out.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("# "):
            out.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.strip():
            out.append(f"<p>{html.escape(line)}</p>")
    flush_table()
    return f'<html lang="he" dir="rtl"><head><meta charset="utf-8"><style>{CSS}</style></head><body>{"".join(out)}</body></html>'


def main() -> int:
    for md in sorted(SRC.glob("*.md")):
        pdf = OUT / (md.stem + ".pdf")
        with tempfile.TemporaryDirectory() as tmp:
            page = Path(tmp) / "page.html"
            page.write_text(md_to_html(md.read_text(encoding="utf-8")), encoding="utf-8")
            subprocess.run(
                [str(EDGE), "--headless", "--disable-gpu", "--no-pdf-header-footer",
                 f"--user-data-dir={Path(tmp) / 'profile'}", f"--print-to-pdf={pdf}", page.as_uri()],
                check=True, capture_output=True, timeout=120,
            )
        print("נוצר:", pdf.name, pdf.stat().st_size, "בתים")
    return 0


if __name__ == "__main__":
    sys.exit(main())
