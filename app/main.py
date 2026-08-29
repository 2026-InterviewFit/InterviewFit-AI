from urllib.request import Request

from fastapi import FastAPI

from app.lifespan import lifespan
from app.schemas import AnalyzeTask, AnalyzeResult
from app.inference import process_gpu_inference


app = FastAPI(title="STAR 면접 답변 분석 서버", lifespan=lifespan)

@app.post("/analyze", response_model=AnalyzeResult)
async def receive_analyze_request(task: AnalyzeTask, request: Request):
    return await process_gpu_inference(
        task=task,
        model=request.app.state.model,
        tokenizer=request.app.state.tokenizer
    )