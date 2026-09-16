from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    VLLM_BASE_URL: str = "http://localhost:8020"
    VLLM_MODEL_ID: str

    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-3.5-flash"

    class Config:
        env_file = ".env"


settings = Settings()


VLLM_BASE_URL = settings.VLLM_BASE_URL
VLLM_MODEL_ID = settings.VLLM_MODEL_ID

GEMINI_API_KEY = settings.GEMINI_API_KEY
GEMINI_MODEL = settings.GEMINI_MODEL