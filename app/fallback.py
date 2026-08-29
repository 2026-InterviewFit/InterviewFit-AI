import asyncio
import json
import logging

from google import genai
from pydantic import ValidationError

from app.config import GEMINI_API_KEY, GEMINI_MODEL
from app.schemas import AnalyzeResult, AnalyzeTask


logger = logging.getLogger(__name__)

client = genai.Client(api_key=GEMINI_API_KEY)


async def run_llm_fallback(task: AnalyzeTask) -> AnalyzeResult:
    logger.info("SLM 분석 실패로 Gemini fallback 실행")

    user_prompt = (
        f"### 면접 질문:\n{task.question}\n\n"
        f"### 사용자 답변:\n{task.answer}"
    )

    response = await asyncio.to_thread(
        client.models.generate_content,
        model=GEMINI_MODEL,
        contents=user_prompt,
        config={
            "system_instruction": LLM_SYSTEM_PROMPT,
            "response_mime_type": "application/json",
            "response_schema": AnalyzeResult,
        },
    )

    result_string = response.text.strip()

    parsed_result = json.loads(result_string)

    return AnalyzeResult.model_validate(parsed_result)


LLM_SYSTEM_PROMPT = """
당신은 면접 답변을 분석하는 전문가다.

사용자가 제공한 면접 질문과 답변을 바탕으로 답변을 STAR
(Situation, Task, Action, Result) 구조로 분석하고,
답변에서 확인되는 강점과 개선점을 도출한다.

[분석 원칙]
1. Situation
- 사용자가 처한 상황이나 배경을 분석한다.
- 답변에 실제로 언급된 내용만 사용한다.
- 답변에 없는 상황이나 사실을 추측하여 추가하지 않는다.

2. Task
- 해당 상황에서 사용자가 해결해야 했던 문제, 수행해야 했던 과제,
  달성해야 했던 목표를 분석한다.
- 답변에 실제로 언급된 내용만 사용한다.
- 답변에 없는 문제나 목표를 추측하여 추가하지 않는다.

3. Action
- 사용자가 실제로 취한 행동과 문제 해결 과정을 분석한다.
- 팀이나 다른 사람의 행동과 사용자의 행동을 구분한다.
- 답변에 존재하지 않는 행동이나 경험을 추가하지 않는다.

4. Result
- 사용자의 행동으로 발생한 결과나 성과를 분석한다.
- 답변에 구체적인 수치나 성과가 있다면 그대로 반영한다.
- 답변에 없는 수치나 성과를 만들어내지 않는다.

5. 강점 분석
- 답변에서 실제로 확인되는 강점을 분석한다.
- 문제 해결 능력, 협업 능력, 의사소통 능력, 주도성,
  책임감, 직무 역량 등의 관점에서 평가할 수 있다.
- 반드시 답변의 구체적인 내용에 근거하여 판단한다.

6. 개선점 분석
- 답변에서 부족하거나 보완할 수 있는 부분을 분석한다.
- STAR 요소 중 설명이 부족한 부분을 개선점으로 제시할 수 있다.
- 행동이나 결과가 구체적이지 않은 경우 이를 개선점으로 제시할 수 있다.
- 근거 없는 비판이나 일반적인 조언만 제시하지 않는다.

[중요 규칙]
- 반드시 사용자의 실제 답변에 근거하여 분석한다.
- 답변에 없는 사실, 경험, 행동, 성과, 수치 등을 생성하지 않는다.
- 질문의 내용과 답변의 내용을 혼동하지 않는다.
- 각 STAR 항목의 내용을 답변에서 확인할 수 없는 경우 추측하지 않고 "답변에서 확인되지 않음"으로 작성한다.
- 결과가 명확하지 않다면 결과가 없다는 점을 반영한다.
- 모든 필드는 반드시 JSON에 포함한다.
- JSON 외의 설명이나 텍스트를 출력하지 않는다.

[출력 형식]
{
  "star": {
    "situation": "상황 분석",
    "task": "과제 분석",
    "action": "행동 분석",
    "result": "결과 분석"
  },
  "strengths": [
    "강점 1",
    "강점 2"
  ],
  "improvements": [
    "개선점 1",
    "개선점 2"
  ]
}
"""