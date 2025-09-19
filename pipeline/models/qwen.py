import torch
from vllm import LLM, SamplingParams
from typing import List, Tuple, Dict
import time
from tqdm import tqdm
import gc

_vllm_model = None
_model_initialized = False

_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"

MAX_BATCH_SIZE = 16
MAX_SEQUENCE_LENGTH = 8192


def get_model():
    global _vllm_model, _model_initialized

    if not _model_initialized:
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        tp_size = min(gpu_count, 1) if gpu_count > 0 else 1

        _vllm_model = LLM(
            model=_MODEL_ID,
            dtype="bfloat16",
            tensor_parallel_size=tp_size,
            trust_remote_code=True,
            enforce_eager=False,
            gpu_memory_utilization=0.90,
            max_model_len=MAX_SEQUENCE_LENGTH,
            quantization=None,
            enable_chunked_prefill=True,
        )
        _model_initialized = True
        print(
            f"Qwen2.5-14B-Instruct Model Loaded with vLLM (Tensor Parallel Size: {tp_size})"
        )

    return _vllm_model


def clear_model_memory():
    global _vllm_model, _model_initialized

    if _model_initialized:

        del _vllm_model
        _vllm_model = None
        _model_initialized = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        print("Gemma model memory and cache cleared.")


def prepare_prompts(items: List[Dict]) -> List[str]:

    prompts = [
        f"<|im_start|>system\nYou are a professional translator who translates text accurately from {item['source_language']} to {item['target_language']}. Provide only the translation without explanations or additional text.<|im_end|>\n<|im_start|>user\nTranslate the following {item['source_language']} text to {item['target_language']}:\n\n{item['src']}<|im_end|>\n<|im_start|>assistant\n"
        for item in items
    ]

    return prompts


def translate_batch(items: List[Dict], batch_size: int = MAX_BATCH_SIZE) -> List[Dict]:

    model = get_model()
    prompts = prepare_prompts(items)
    sampling_params = SamplingParams(
        temperature=0.3,
        top_p=0.9,
        max_tokens=1024,
        stop=["<|im_end|>", "<|im_start|>"],
    )

    outputs = model.generate(prompts, sampling_params)
    all_translations = []
    for i, output in enumerate(outputs):
        original_item = items[i]
        generated_text = output.outputs[0].text.strip()

        output_item = original_item.copy()
        output_item["mt"] = generated_text
        all_translations.append(output_item)

    return all_translations


def translate_batch_with_timing(
    items: List[Dict], batch_size: int = MAX_BATCH_SIZE
) -> Tuple[List[Dict], float]:

    clear_model_memory()
    start_time = time.time()
    translations = translate_batch(items, batch_size)
    end_time = time.time()
    clear_model_memory()
    return translations, end_time - start_time
