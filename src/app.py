"""צ'אט בעברית ב-Streamlit. הפעלה: streamlit run src/app.py"""
import streamlit as st

from models import MissingApiKey
from qlog import log_error, log_trace
from rag import build_pipeline

st.set_page_config(page_title="שביל החלב משחקים", page_icon="🎲", layout="centered")

st.markdown(
    """
    <style>
    .stApp, [data-testid="stChatMessage"], [data-testid="stChatInput"] textarea,
    [data-testid="stMarkdownContainer"] { direction: rtl; text-align: right; }
    </style>
    """,
    unsafe_allow_html=True,
)

MSG_NO_KEY = "המערכת עדיין לא הוגדרה: חסר מפתח גישה. יש להוסיף אותו בקובץ ‎.env‎ ולהפעיל את האפליקציה מחדש."
MSG_FAILED = "מצטערים, לא הצלחנו לענות כרגע. אפשר לנסות שוב בעוד רגע."


@st.cache_resource(show_spinner="טוען את המערכת...")
def load_pipeline():
    return build_pipeline()


def answer_question(question: str) -> str:
    try:
        pipe = load_pipeline()
        tr = pipe.ask(question)
    except MissingApiKey as e:
        log_error("app", e)
        return MSG_NO_KEY
    except Exception as e:  # כשל רשת, מפתח לא תקין, מכסה וכו'. ללא פרטים טכניים למשתמש
        log_error("app", e)
        return MSG_FAILED
    log_trace(tr, source="app", embed_model=pipe.embed.model_name)
    return tr.answer


st.title("🎲 שביל החלב משחקים")
st.caption("אפשר לשאול על חוקי המשחקים, מחירים, החזרות ומשלוחים, ושירות לקוחות.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if question := st.chat_input("כתבו כאן את השאלה שלכם"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("מחפש תשובה..."):
            reply = answer_question(question)
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
