from langchain_ollama import ChatOllama

from .config import settings

llm = ChatOllama(
    model=settings.ollama_model,
    base_url=settings.ollama_base_url,
    num_ctx=settings.ollama_num_ctx,
    temperature=0,
    # Ignored by local Ollama; used if we switch back to the cloud API.
    client_kwargs={"headers": {"Authorization": f"Bearer {settings.ollama_api_key}"}},
)