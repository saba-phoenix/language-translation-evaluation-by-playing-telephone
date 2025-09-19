import torch
import ctranslate2
import transformers
from typing import List, Tuple, Dict
import time
from tqdm import tqdm
import json
import os
import gc


_nllb_translator = None
_nllb_tokenizer = None
_model_initialized = False


_MODEL_ID = "facebook/nllb-200-distilled-600M"
_CT2_MODEL_DIR = "nllb-200-distilled-600M-ct2-float32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_INDEX = 0  # adjust if using multiple GPUs


MAX_BATCH_SIZE = 128
MAX_LENGTH = 512

LANG_CODES_PATH = os.path.join(
    os.path.dirname(__file__), "..", "utils", "data_utils", "ISO Language Codes.json"
)
with open(LANG_CODES_PATH, "r", encoding="utf-8") as f:
    LANGUAGE_CODES = json.load(f)


def convert_model_if_needed():

    if not os.path.exists(_CT2_MODEL_DIR):
        print(f"Converting {_MODEL_ID} to CTranslate2 format...")
        os.system(
            f"ct2-transformers-converter --model {_MODEL_ID} --output_dir {_CT2_MODEL_DIR}"
        )
        print("Model conversion completed.")


def get_model():

    global _nllb_translator, _nllb_tokenizer, _model_initialized

    if not _model_initialized:
        print("Loading NLLB Model with CTranslate2...")
        convert_model_if_needed()
        _nllb_tokenizer = transformers.AutoTokenizer.from_pretrained(_MODEL_ID)

        _nllb_translator = ctranslate2.Translator(
            _CT2_MODEL_DIR,
            device=DEVICE,
            device_index=DEVICE_INDEX,
            compute_type="float32",
            inter_threads=4,  # number of parallel translations
            intra_threads=0,
            max_queued_batches=1000,
        )

        _model_initialized = True
        print("NLLB Model Loaded with CTranslate2.")

    return _nllb_tokenizer, _nllb_translator


def clear_model_memory():

    global _nllb_translator, _nllb_tokenizer, _model_initialized

    if _model_initialized:
        print("Clearing model memory and cache...")

        del _nllb_translator
        del _nllb_tokenizer

        _model_initialized = False
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        print("Model memory and cache cleared.")


def translate_batch(items: List[Dict], batch_size: int = MAX_BATCH_SIZE) -> List[Dict]:

    tokenizer, translator = get_model()

    language_pairs = {}
    for item in items:
        src_lang = LANGUAGE_CODES[item["source_language"]]["Flores_code"]
        tgt_lang = LANGUAGE_CODES[item["target_language"]]["Flores_code"]
        pair_key = (src_lang, tgt_lang)

        if pair_key not in language_pairs:
            language_pairs[pair_key] = []
        language_pairs[pair_key].append(item)

    all_translations = []

    for pair_key, pair_items in language_pairs.items():
        src_lang, tgt_lang = pair_key

        for i in tqdm(range(0, len(pair_items), batch_size)):
            batch_items = pair_items[i : i + batch_size]

            tokenizer.src_lang = src_lang
            source_texts = [item["src"] for item in batch_items]
            encoded_sources = [
                tokenizer.convert_ids_to_tokens(
                    tokenizer.encode(text, add_special_tokens=True)
                )
                for text in source_texts
            ]

            target_prefixes = [[tgt_lang]] * len(encoded_sources)

            results = translator.translate_batch(
                encoded_sources,
                target_prefix=target_prefixes,
                max_batch_size=batch_size,
                batch_type="examples",
                max_input_length=MAX_LENGTH,
                beam_size=4,
                num_hypotheses=1,
                length_penalty=0.6,
                return_scores=False,
                use_vmap=True,
                replace_unknowns=True,
                max_decoding_length=MAX_LENGTH,
            )

            for j, result in enumerate(results):
                original_item = batch_items[j]
                translation_tokens = result.hypotheses[0][1:]
                translation = tokenizer.decode(
                    tokenizer.convert_tokens_to_ids(translation_tokens),
                    skip_special_tokens=True,
                )

                output_item = original_item.copy()
                output_item["mt"] = translation
                all_translations.append(output_item)

    return all_translations


def translate_batch_with_timing(
    items: List[Dict], batch_size: int = MAX_BATCH_SIZE
) -> Tuple[List[Dict], float]:

    clear_model_memory()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()
    translations = translate_batch(items, batch_size)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()
    clear_model_memory()

    return translations, end_time - start_time
