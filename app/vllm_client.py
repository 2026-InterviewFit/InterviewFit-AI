import logging

from openai import AsyncOpenAI

from app.config import VLLM_BASE_URL, VLLM_MODEL_ID


logger = logging.getLogger(__name__)

client = AsyncOpenAI(
    base_url=f"{VLLM_BASE_URL}",
    api_key="EMPTY",
)


async def generate(messages: list[dict], max_retries: int = 1):
    for attempt in range(max_retries + 1):
        try:
            response = await client.chat.completions.create(
                model=VLLM_MODEL_ID,
                messages=messages,
                max_tokens=512,
                temperature=0,
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": False,
                    },
                },
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries:
                logger.warning(
                    f"vLLM 통신 오류 발생. 재시도 "
                    f"({attempt + 1}/{max_retries})"
                )
                continue
            raise e
