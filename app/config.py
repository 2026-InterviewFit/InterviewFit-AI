from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Model Configuration
    BASE_MODEL_ID: str = "Qwen/Qwen3-4B"
    LORA_MODEL_PATH: str

    class Config:
        env_file = ".env"


settings = Settings()


BASE_MODEL_ID = settings.BASE_MODEL_ID
LORA_MODEL_PATH = settings.LORA_MODEL_PATH