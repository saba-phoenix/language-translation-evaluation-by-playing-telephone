import torch
from vllm import LLM, SamplingParams
from vllm.utils import random_uuid
from typing import List, Tuple, Dict
import time
from tqdm import tqdm
import gc
import traceback

# globals for lazy loading
_vllm_model = None
_model_initialized = False

_MODEL_ID = "google/gemma-3-12b-it"

MAX_SEQUENCE_LENGTH = 4096


def get_model():
    global _vllm_model, _model_initialized

    if not _model_initialized:

        # check GPU availability
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
            enable_chunked_prefill=True,
        )
        _model_initialized = True
        print(f"Gemma-3-12B Model Loaded with vLLM (Tensor Parallel Size: {tp_size})")

    return _vllm_model


def clear_model_memory():
    global _vllm_model, _model_initialized

    if _model_initialized:
        print("Clearing Gemma model memory and cache...")

        del _vllm_model
        _vllm_model = None
        _model_initialized = False
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        print("Gemma model memory and cache cleared.")


def prepare_prompt(item: Dict) -> str:

    prompt = f"<start_of_turn>user\nYou are a professional translator working on an important project.\n\nTranslate the following text from {item['source_language']} to {item['target_language']}.\n\n"
    prompt += f"IMPORTANT RULES:\n"
    prompt += f"1. Your response must contain ONLY the translated text\n"
    prompt += f"2. Do not include any explanations, comments, or notes\n"
    prompt += f"3. Translate the full text completely\n"
    prompt += f"4. Never leave the translation empty\n\n"
    prompt += f"Text to translate:\n{item['src']}<end_of_turn>\n<start_of_turn>model\n"

    return prompt


def translate_batch(items: List[Dict], batch_size: int = None) -> List[Dict]:
    """
    Translates a batch of items using Gemma-3-12B with vLLM.
    Processes all items in a single batch without any internal batching.
    """
    if not items:
        return []

    model = get_model()

    prompts = [prepare_prompt(item) for item in items]

    # optimized sampling parameters
    sampling_params = SamplingParams(
        temperature=0.3,
        top_p=0.95,
        top_k=40,
        max_tokens=1024,
        stop=["\n\n", "<end_of_turn>", "<start_of_turn>"],
        repetition_penalty=1.2,
        frequency_penalty=0.5,
    )

    outputs = model.generate(prompts, sampling_params)

    all_translations = []
    for i, output in enumerate(outputs):
        original_item = items[i]
        generated_text = output.outputs[0].text.strip()

        # provide a fallback if the translation is empty
        if (
            not generated_text or len(generated_text) < 2
        ):  # consider very short outputs as empty too
            print(f"Warning: Empty translation for item {i}. Using fallback.")
            if original_item["source_language"] == original_item["target_language"]:
                generated_text = original_item["src"]
            else:
                generated_text = (
                    f"[Translation placeholder for: {original_item['src'][:30]}...]"
                )

        output_item = original_item.copy()
        output_item["mt"] = generated_text
        all_translations.append(output_item)

    return all_translations


def translate_batch_with_retry(items: List[Dict], max_retries: int = 3) -> List[Dict]:
    """
    Translates a batch of items with full job retry logic.
    """
    for retry in range(max_retries):
        try:
            return translate_batch(items)
        except Exception as e:
            error_msg = (
                f"Error during translation (attempt {retry+1}/{max_retries}): {str(e)}"
            )
            print(error_msg)
            print(f"Stack trace: {traceback.format_exc()}")

            clear_model_memory()

            # if last retry, raise the exception
            if retry == max_retries - 1:
                print(f"Job failed after {max_retries} attempts. Last error: {str(e)}")
                raise

    return []


def translate_batch_with_timing(
    items: List[Dict],
    batch_size: int = None,
    clear_cache: bool = True,
    is_last: bool = False,
    max_retries: int = 5,
) -> Tuple[List[Dict], float]:

    clear_model_memory()

    start_time = time.time()

    for job_retry in range(max_retries):
        try:
            print(
                f"Processing all {len(items)} items at once (job attempt {job_retry+1}/{max_retries})..."
            )
            translations = translate_batch(items)

            # if we got here, the job was successful
            end_time = time.time()
            print(f"Job completed successfully in {end_time - start_time:.2f} seconds")

            clear_model_memory()

            return translations, end_time - start_time

        except Exception as e:
            error_msg = f"Job error on attempt {job_retry+1}/{max_retries}: {str(e)}"
            print(error_msg)
            print(f"Stack trace: {traceback.format_exc()}")

            clear_model_memory()

            if job_retry == max_retries - 1:
                print(
                    f"All {max_retries} job attempts failed. Creating fallback translations."
                )

                # fallback translations
                fallback_translations = []
                for item in items:
                    fallback_item = item.copy()
                    if item["source_language"] == item["target_language"]:
                        fallback_item["mt"] = item["src"]
                    else:
                        fallback_item["mt"] = (
                            f"[Translation failed after {max_retries} attempts]"
                        )
                    fallback_translations.append(fallback_item)

                end_time = time.time()

                clear_model_memory()

                return fallback_translations, end_time - start_time

    return [], 0.0
