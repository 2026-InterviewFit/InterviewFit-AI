from pydantic import BaseModel


# Request Schema
class AnalyzeTask(BaseModel):
    question: str
    answer: str


# STAR Analysis Schema
class StarAnalyze(BaseModel):
    situation: str
    task: str
    action: str
    result: str

# Response Schema
class AnalyzeResult(BaseModel):
    star: StarAnalyze
    strengths: list[str]
    improvements: list[str]