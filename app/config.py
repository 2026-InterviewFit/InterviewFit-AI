from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Model Configuration
    BASE_MODEL_ID: str = "Qwen/Qwen3-4B"
    LORA_MODEL_PATH: str

    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-3.5-flash"

    class Config:
        env_file = ".env"


settings = Settings()


BASE_MODEL_ID = settings.BASE_MODEL_ID
LORA_MODEL_PATH = settings.LORA_MODEL_PATH

GEMINI_API_KEY = settings.GEMINI_API_KEY
GEMINI_MODEL = settings.GEMINI_MODEL