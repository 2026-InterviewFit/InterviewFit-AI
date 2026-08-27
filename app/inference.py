import asyncio
import json
import logging

import torch

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


async def process_gpu_inference(task: AnalyzeTask) -> AnalyzeResult:
    async with gpu_lock:
        try:
            # 1. Prompt 생성
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

            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # 2. Tokenize
            inputs = tokenizer(
                prompt_text,
                return_tensors="pt",
            ).to("cuda")

            # 3. GPU Inference
            outputs = await asyncio.to_thread(
                generate_response,
                inputs,
            )

            # 4. 생성된 부분만 추출
            generated_tokens = outputs[
                :,
                inputs.input_ids.shape[1]:,
            ]

            result_string = tokenizer.decode(
                generated_tokens[0],
                skip_special_tokens=True,
            )

            # 5. JSON 정리
            result_string = clean_json(result_string)

            # 6. JSON Parsing
            parsed_result = json.loads(result_string)

            # 7. Pydantic 검증 및 결과 리턴
            return AnalyzeResult.model_validate(parsed_result)
        except Exception as e:
            logger.exception("분석 실패: %s",e)
            raise


def generate_response(inputs):
    with torch.inference_mode():
        return model.generate(
            **inputs,
            max_new_tokens=1280,
            do_sample=False,
            repetition_penalty=1.0,
        )