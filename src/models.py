"""יצירת מודלי LlamaIndex לפי ההגדרות."""
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from config import EMBED_MODEL, LLM_MODEL, get_api_key


class MissingApiKey(RuntimeError):
    pass


def _key() -> str:
    key = get_api_key()
    if not key:
        raise MissingApiKey("חסר OPENAI_API_KEY בקובץ .env")
    return key


def get_embed_model(model: str | None = None) -> OpenAIEmbedding:
    return OpenAIEmbedding(model=model or EMBED_MODEL, api_key=_key())


def get_llm(temperature: float = 0.0, json_mode: bool = False) -> OpenAI:
    extra = {"response_format": {"type": "json_object"}} if json_mode else {}
    return OpenAI(model=LLM_MODEL, api_key=_key(), temperature=temperature, additional_kwargs=extra)
