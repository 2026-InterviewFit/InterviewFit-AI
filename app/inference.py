import json
import logging
import httpx

from pydantic import ValidationError

from app.fallback import run_llm_fallback
from app.schemas import AnalyzeTask, AnalyzeResult
from app.utils import clean_json
from app.vllm_client import generate


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "면접 질문과 사용자의 답변을 바탕으로 답변을 STAR 구조로 분석하고, "
    "답변에서 확인되는 강점과 개선점을 분석한다. "
    "분석 결과는 지정된 JSON 형식으로 출력한다."
)


async def process_inference(task: AnalyzeTask) -> AnalyzeResult:
    try:
        messages = build_messages(task)
        result_string = await generate(messages)
        return parse_response(result_string)
    except json.JSONDecodeError:
        logger.exception("SLM JSON 파싱 실패")
        return await run_llm_fallback(task)
    except ValidationError:
        logger.exception("SLM 응답 형식 검증 실패")
        return await run_llm_fallback(task)
    except (httpx.TimeoutException, httpx.ConnectError):
        logger.exception("vLLM 통신 오류")
        return await run_llm_fallback(task)
    except Exception:
        logger.exception("분석 중 예상하지 못한 오류")
        raise


def build_messages(task: AnalyzeTask) -> list[dict]:
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                f"### 질문:\n{task.question}\n\n"
                f"### 답변:\n{task.answer}"
            ),
        },
    ]


def parse_response(result_string: str) -> AnalyzeResult:
    result_string = clean_json(result_string)
    parsed_result = json.loads(result_string)
    return AnalyzeResult.model_validate(parsed_result)