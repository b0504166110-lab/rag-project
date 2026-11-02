"""בדיקת הממשק בלי דפדפן. שימוש: python scripts/test_app.py [ok|nokey|badkey]

ok     שאלה רגילה (קריאה אמיתית ל-API)
nokey  מצב ללא מפתח
badkey מפתח לא תקין, כלומר הקריאה נכשלת
"""
import os
import sys
from pathlib import Path

mode = sys.argv[1] if len(sys.argv) > 1 else "ok"
if mode == "nokey":
    os.environ["OPENAI_API_KEY"] = ""
elif mode == "badkey":
    os.environ["OPENAI_API_KEY"] = "sk-invalid-key-for-test"

from streamlit.testing.v1 import AppTest  # noqa: E402

app = Path(__file__).resolve().parent.parent / "src" / "app.py"
at = AppTest.from_file(str(app), default_timeout=180)
at.run()
at.chat_input[0].set_value("כמה עולה עמק הדבורים?").run()
print("חריגות:", [e.value for e in at.exception])
for m in at.chat_message:
    print(m.name, "|", m.markdown[0].value if m.markdown else "")
page = " ".join(x.value for x in at.markdown)
leaks = [w for w in ["sk-", "Error", "Traceback", "ציון", "מקור", "סינון", "טוקנ"] if w in page]
print("מילים טכניות בעמוד:", leaks or "אין")
