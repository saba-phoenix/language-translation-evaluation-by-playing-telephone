import torch
from vllm import LLM, SamplingParams
from vllm.utils import random_uuid
from typing import List, Dict, Any, Tuple
import time
from tqdm import tqdm


MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
MAX_SEQUENCE_LENGTH = 2048
MAX_BATCH_SIZE = 4

_vllm_model = None


def get_model():
    global _vllm_model

    if _vllm_model is None:

        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        print(f"Found {gpu_count} GPUs")

        _vllm_model = LLM(
            model=MODEL_ID,
            dtype="bfloat16",
            tensor_parallel_size=gpu_count,
            trust_remote_code=True,
            enforce_eager=True,
            gpu_memory_utilization=0.8,
            max_model_len=MAX_SEQUENCE_LENGTH,
            enable_lora=False,
            quantization="awq",
            max_num_batched_tokens=4096,
        )

        print(f"{MODEL_ID} loaded successfully")

    return _vllm_model


def prepare_prompts(items: List[Dict]) -> List[str]:

    system_prompt = "You are a translator. Translate the text directly without any additional text or explanations."

    prompts = []
    for item in items:
        user_message = f"Translate this text from {item['source_language']} to {item['target_language']}: {item['src']}"
        prompt = f"<s>[INST] {system_prompt}\n\n{user_message} [/INST]"
        prompts.append(prompt)

    return prompts


def translate_batch(items: List[Dict], batch_size: int = MAX_BATCH_SIZE) -> List[Dict]:

    model = get_model()

    all_translations = []

    for i in tqdm(range(0, len(items), batch_size)):
        batch_items = items[i : i + batch_size]

        prompts = prepare_prompts(batch_items)
        request_ids = [random_uuid() for _ in range(len(batch_items))]

        # mapping from request ID to original item
        id_to_item = {req_id: item for req_id, item in zip(request_ids, batch_items)}

        max_tokens_list = [
            min(256, max(len(item["src"].split()) * 2, 64)) for item in batch_items
        ]
        max_tokens = max(max_tokens_list)

        sampling_params = SamplingParams(
            temperature=0.2,
            top_p=0.95,
            max_tokens=max_tokens,
            stop=["</s>", "[INST]", "<s>"],
            use_beam_search=False,
        )

        print(
            f"Generating translations for batch {i//batch_size + 1} ({len(batch_items)} items)..."
        )
        outputs = model.generate(prompts, sampling_params, request_ids=request_ids)

        for output in outputs:
            original_item = id_to_item[output.request_id]
            generated_text = output.outputs[0].text.strip()

            translation = generated_text
            translation = translation.replace("[/INST]", "").strip()
            translation = translation.replace("[INST]", "").strip()
            translation = translation.replace("VENDOR:", "").strip()
            translation = translation.replace(
                "Sure, here's the translation:", ""
            ).strip()
            translation = translation.replace('"', "").strip()

            if translation.startswith("Translation:"):
                translation = translation[len("Translation:") :].strip()

            output_item = original_item.copy()
            output_item["mt"] = translation
            all_translations.append(output_item)

        torch.cuda.empty_cache()

    return all_translations


def translate_batch_with_timing(
    items: List[Dict], batch_size: int = MAX_BATCH_SIZE
) -> Tuple[List[Dict], float]:

    start_time = time.time()
    translations = translate_batch(items, batch_size)
    end_time = time.time()

    return translations, end_time - start_time
