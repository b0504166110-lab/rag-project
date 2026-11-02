"""בדיקה שהטקסט שנחלץ מקובצי ה-PDF זהה למקור שלהם (תוכן וסדר), בלי קריאות API."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from loader import read_text  # noqa: E402


def norm(t: str) -> str:
    t = re.sub(r"^#+\s*", "", t, flags=re.M)
    return re.sub(r"\s+", " ", t).strip()


def main() -> int:
    bad = 0
    for pdf in sorted((ROOT / "data").glob("*.pdf")):
        src = (ROOT / "pdf_src" / (pdf.stem + ".md")).read_text(encoding="utf-8")
        got, want = norm(read_text(pdf)), norm(src)
        ok = got == want
        bad += not ok
        print(("תקין  " if ok else "שגוי  ") + pdf.name)
        if not ok:
            for i, (a, b) in enumerate(zip(got, want)):
                if a != b:
                    print("  ראשון שונה במיקום", i, "| נחלץ:", got[max(0, i - 30):i + 30], "| מקור:", want[max(0, i - 30):i + 30])
                    break
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
