"""Model loading and chat-template rendering."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from pilot.config import ModelSpec


def load(spec: ModelSpec):
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, padding_side="left")
    quantization = (
        BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.bfloat16)
        if spec.load_in_4bit else None
    )
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id, dtype=torch.bfloat16, device_map="cuda", quantization_config=quantization,
    )
    model.eval()
    return model, tokenizer


def render(tokenizer, user_message: str, thinking: bool = True) -> str:
    """Single-turn chat prompt, ending right where the assistant starts writing."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message}],
        tokenize=False, add_generation_prompt=True, enable_thinking=thinking,
    )
