from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ollama_api_key: str
    ollama_base_url: str = "https://ollama.com"
    ollama_model: str = "gemma4:31b"
    django_api_url: str = "http://web:8000"
    ollama_num_ctx: int = 8192
    # redis_url: str = "redis://redis:6379/1"
    agent_db_url: str
    app_timezone: str = "Asia/Karachi"

    model_config = SettingsConfigDict(extra="ignore")


settings = Settings()