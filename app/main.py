from fastapi import Request, FastAPI

from app.schemas import AnalyzeTask, AnalyzeResult
from app.inference import process_inference


app = FastAPI(title="STAR 면접 답변 분석 서버")


@app.post("/analyze", response_model=AnalyzeResult)
async def receive_analyze_request(task: AnalyzeTask, request: Request):
    return await process_inference(task=task)