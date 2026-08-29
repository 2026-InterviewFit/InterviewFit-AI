from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.model import load_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, tokenizer = load_model()

    app.state.model = model
    app.state.tokenizer = tokenizer

    yield

    # 서버 종료 시 모델 정리
    del app.state.model
    del app.state.tokenizer