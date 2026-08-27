import logging

import torch

from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from app.config import BASE_MODEL_ID, LORA_MODEL_PATH


logger = logging.getLogger(__name__)


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

logger.info("Qwen3-4B 모델 로드 시작")

tokenizer = AutoTokenizer.from_pretrained(
    LORA_MODEL_PATH,
)

logger.info("Tokenizer 로드 완료")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map={"": 0},
    attn_implementation="sdpa",
)

logger.info("Base model 로드 완료")

model = PeftModel.from_pretrained(
    base_model,
    LORA_MODEL_PATH,
)

model.eval()

logger.info("Qwen3-4B + LoRA 모델 로드 완료")