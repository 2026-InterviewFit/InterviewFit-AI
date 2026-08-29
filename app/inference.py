import asyncio
import json
import logging

import torch

from pydantic import ValidationError

from app.fallback import run_llm_fallback
from app.model import model, tokenizer
from app.schemas import AnalyzeTask, AnalyzeResult
from app.utils import clean_json


logger = logging.getLogger(__name__)

# GPU Lock : 동시에 여러 inference가 GPU를 사용하지 않도록 방지
gpu_lock = asyncio.Lock()


SYSTEM_PROMPT = (
    "면접 질문과 사용자의 답변을 바탕으로 답변을 STAR 구조로 분석하고, "
    "답변에서 확인되는 강점과 개선점을 분석한다. "
    "분석 결과는 지정된 JSON 형식으로 출력한다."
)


async def process_gpu_inference(
    task: AnalyzeTask,
    model,
    tokenizer
) -> AnalyzeResult:
    async with gpu_lock:
        try:
            prompt_text = build_prompt(task, tokenizer)

            inputs, outputs = await run_generation(
                prompt_text,
                model,
                tokenizer,
            )

            return parse_response(
                outputs,
                inputs,
                tokenizer,
            )
        except json.JSONDecodeError:
            logger.exception("SLM JSON 파싱 실패")
            return await run_llm_fallback(task)
        except ValidationError:
            logger.exception("SLM 응답 형식 검증 실패")
            return await run_llm_fallback(task)
        except RuntimeError:
            logger.exception("SLM 추론 중 런타임 오류")
            raise
        except Exception:
            logger.exception("분석 중 예상하지 못한 오류")
            raise


def build_prompt(task: AnalyzeTask, tokenizer) -> str:
    messages = [
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

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


async def run_generation(prompt_text: str, model, tokenizer):
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
    ).to("cuda")

    outputs = await asyncio.to_thread(
        generate_response,
        model,
        inputs,
    )

    return inputs, outputs


def generate_response(model, inputs):
    with torch.inference_mode():
        return model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.1,
        )


def parse_response(outputs, inputs, tokenizer) -> AnalyzeResult:
    generated_tokens = outputs[
        :,
        inputs.input_ids.shape[1]:,
    ]

    result_string = tokenizer.decode(
        generated_tokens[0],
        skip_special_tokens=True,
    )

    result_string = clean_json(result_string)

    parsed_result = json.loads(result_string)

    return AnalyzeResult.model_validate(parsed_result)