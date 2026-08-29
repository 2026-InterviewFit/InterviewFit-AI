from fastapi import FastAPI

from app.schemas import AnalyzeTask, AnalyzeResult
from app.inference import process_gpu_inference


app = FastAPI(title="STAR 면접 답변 분석 서버")

# Analyze API
@app.post("/analyze", response_model=AnalyzeResult)
async def receive_analyze_request(task: AnalyzeTask):
    return await process_gpu_inference(task)