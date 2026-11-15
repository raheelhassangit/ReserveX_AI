from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ollama_api_key: str
    ollama_base_url: str = "https://ollama.com"
    ollama_model: str = "gemma4:31b"
    django_api_url: str = "http://web:8000"

    model_config = SettingsConfigDict(extra="ignore")


settings = Settings()