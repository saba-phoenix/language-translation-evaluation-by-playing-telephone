import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple, Dict
import time
from tqdm import tqdm
import logging

_suzume_model = None
_suzume_tokenizer = None
_model_initialized = False

_MODEL_ID = "lightblue/suzume-llama-3-8B-multilingual"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_BATCH_SIZE = 16
MAX_SEQUENCE_LENGTH = 2048


def get_model():
    global _suzume_model, _suzume_tokenizer, _model_initialized

    if not _model_initialized:
        _suzume_tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
        _suzume_tokenizer.pad_token = _suzume_tokenizer.eos_token
        _suzume_tokenizer.padding_side = "left"

        _suzume_model = AutoModelForCausalLM.from_pretrained(
            _MODEL_ID,
            torch_dtype=torch.bfloat16,
        ).to(DEVICE)
        _suzume_model.config.pad_token_id = _suzume_tokenizer.pad_token_id

        _model_initialized = True
        print("Suzume Multilingual Model Loaded.")

    return _suzume_tokenizer, _suzume_model


def prepare_batch(
    items: List[Dict],
) -> Tuple[torch.Tensor, List[int]]:

    tokenizer, _ = get_model()

    prompts = [
        f"""
You are a professional translator. Translate the text below from {item['source_language']} to {item['target_language']}. 
Respond with ONLY the translation in natural {item['target_language']}. Do not include any other text.

Text: {item['src']}
Translation: """
        for item in items
    ]

    inputs = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
        return_tensors="pt",
    ).to(DEVICE)

    original_lengths = [len(tokenizer.encode(item["src"])) for item in items]

    return inputs, original_lengths


def translate_batch(items: List[Dict], batch_size: int = MAX_BATCH_SIZE) -> List[Dict]:

    tokenizer, model = get_model()

    all_translations = []
    for i in tqdm(range(0, len(items), batch_size), miniters=50):
        batch_items = items[i : i + batch_size]

        inputs, original_lengths = prepare_batch(batch_items)

        max_input_tokens = max(original_lengths)
        max_new_tokens = min(max_input_tokens * 2, 2048)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
            num_beams=3,
            no_repeat_ngram_size=3,
            length_penalty=0.9,
            early_stopping=True,
            min_length=max_input_tokens,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for item, response in zip(batch_items, responses):

            # clean up the translation
            translation = response.split("Translation:")[-1].strip()
            translation = translation.split("Input (")[0].strip()
            translation = translation.split("Now translate")[0].strip()
            translation = translation.split("Example")[0].strip()
            translation = translation.replace("\n", " ")
            translation = translation.strip("\"'")
            translation = translation.replace("[/INST]", "").strip()
            translation = translation.replace("[INST]", "").strip()

            translation_tokens = len(tokenizer.encode(translation))
            source_tokens = len(tokenizer.encode(item["src"]))
            if translation_tokens < source_tokens * 0.3:
                logging.warning(
                    f"Translation might be incomplete for item {item['id']}. "
                    f"Source tokens: {source_tokens}, "
                    f"Translation tokens: {translation_tokens}"
                    f"Translation: {translation}"
                    f"Response: {response}"
                )

            output_item = item.copy()
            output_item["mt"] = translation
            all_translations.append(output_item)

    return all_translations


def translate_batch_with_timing(
    items: List[Dict], batch_size: int = MAX_BATCH_SIZE
) -> Tuple[List[Dict], float]:

    start_time = time.time()
    translations = translate_batch(items, batch_size)
    end_time = time.time()

    return translations, end_time - start_time
