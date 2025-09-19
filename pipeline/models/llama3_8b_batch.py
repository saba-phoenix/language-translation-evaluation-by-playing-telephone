import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple, Dict
import time
from tqdm import tqdm
import logging

_llama3_model = None
_llama3_tokenizer = None
_model_initialized = False

# Define Model ID and Device
_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_BATCH_SIZE = 16
MAX_SEQUENCE_LENGTH = 1024

LANGUAGE_EXPANSION_FACTORS = {
    # High-resource languages (Latin script)
    "English": 1.0,
    "German": 1.2,
    "French": 1.2,
    "Spanish": 1.2,
    "Italian": 1.2,
    "Portuguese": 1.2,
    "Dutch": 1.2,
    "Swedish": 1.2,
    "Norwegian": 1.2,
    "Danish": 1.2,
    "Finnish": 1.4,
    "Polish": 1.3,
    "Czech": 1.3,
    "Turkish": 1.3,
    # High-resource languages (non-Latin script)
    "Chinese": 0.9,
    "Japanese": 1.5,
    "Korean": 1.3,
    "Russian": 1.3,
    "Arabic": 1.3,
    "Ukrainian": 1.3,
    # Moderate-resource languages
    "Bengali": 4.0,
    "Hindi": 1.4,
    "Urdu": 1.4,
    "Persian": 1.4,
    "Vietnamese": 1.4,
    "Thai": 1.6,
    "Indonesian": 1.3,
    "Malay": 1.3,
    "Catalan": 1.2,
    "Estonian": 1.3,
    "Hebrew": 1.3,
    "Icelandic": 1.3,
    "Kazakh": 1.3,
    "Latvian": 1.3,
    "Lithuanian": 1.3,
    # Low-resource languages
    "Azerbaijani": 1.3,
    "Basque": 1.3,
    "Central Khmer": 1.4,
    "Croatian": 1.3,
    "Gujarati": 1.4,
    "Hausa": 1.3,
    "Kannada": 1.4,
    "Marathi": 1.4,
    "Pashto": 1.4,
    "Swahili": 1.3,
    "Tagalog": 1.3,
    "Tamil": 1.4,
    "Telugu": 1.4,
    "Xhosa": 1.3,
    "Zulu": 1.3,
    # Default factor for unknown languages
    "default": 2.0,
}


def get_expansion_factor(source_language: str, target_language: str) -> float:
    """
    Get the token expansion factor between source and target languages.
    This represents how many more tokens are needed in the target language compared to the source language.
    """
    source_factor = LANGUAGE_EXPANSION_FACTORS.get(
        source_language, LANGUAGE_EXPANSION_FACTORS["default"]
    )
    target_factor = LANGUAGE_EXPANSION_FACTORS.get(
        target_language, LANGUAGE_EXPANSION_FACTORS["default"]
    )

    relative_factor = target_factor / source_factor

    return relative_factor


def get_model():
    """
    Lazy-loads the Llama-3.1-8B model and tokenizer.
    Returns (tokenizer, model), ensuring they are loaded only once.
    """
    global _llama3_model, _llama3_tokenizer, _model_initialized

    if not _model_initialized:
        print("Loading Llama-3.1-8B Model...")
        _llama3_tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
        _llama3_tokenizer.pad_token = _llama3_tokenizer.eos_token
        _llama3_tokenizer.padding_side = "left"

        _llama3_model = AutoModelForCausalLM.from_pretrained(
            _MODEL_ID,
            torch_dtype=torch.bfloat16,
        ).to(DEVICE)

        _llama3_model.config.pad_token_id = _llama3_tokenizer.pad_token_id

        _model_initialized = True
        print("Llama-3.1-8B Model Loaded.")

    return _llama3_tokenizer, _llama3_model


def prepare_batch(
    items: List[Dict],
) -> Tuple[torch.Tensor, List[int]]:
    tokenizer, _ = get_model()

    prompts = [
        f"""<s>[INST] You are a professional translator. Your task is to translate the following text from {item['source_language']} to {item['target_language']}. 
IMPORTANT RULES:
1. You must output ONLY the translation in {item['target_language']}, not in any other language.
2. DO NOT add any explanations, notes, or additional text.
3. DO NOT use any newlines in your output.
4. DO NOT include any English text in your output.
5. DO NOT include any quotes or special characters in your output.

Here are some examples of correct translations:

Example 1:
Input (English to Bengali): "Hello, how are you?"
Output: হ্যালো, আপনি কেমন আছেন?

Example 2:
Input (Hindi to Bengali): "मैं आपसे मिलकर खुश हूं"
Output: আমি আপনাকে দেখে খুশি

Example 3:
Input (English to Bengali): "The weather is beautiful today"
Output: আজ আবহাওয়া খুব সুন্দর

Example 4:
Input (Hindi to Bengali): "मैं कल दिल्ली जा रहा हूं"
Output: আমি কাল দিল্লি যাচ্ছি

Now translate this text:
Input ({item['source_language']} to {item['target_language']}): "{item['src']}"
Output: [/INST]"""
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


def translate_batch(
    items: List[Dict],
) -> List[Dict]:

    tokenizer, model = get_model()

    all_translations = []
    for i in tqdm(range(0, len(items), MAX_BATCH_SIZE)):
        batch_items = items[i : i + MAX_BATCH_SIZE]

        inputs, original_lengths = prepare_batch(batch_items)

        max_input_tokens = max(original_lengths)
        max_expansion = max(
            get_expansion_factor(item["source_language"], item["target_language"])
            for item in batch_items
        )
        max_new_tokens = min(int(max_input_tokens * max_expansion * 1.2), 4096)

        min_expansion = min(
            get_expansion_factor(item["source_language"], item["target_language"])
            for item in batch_items
        )
        min_length = int(max_input_tokens * min_expansion * 0.9)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
            num_beams=3,
            no_repeat_ngram_size=3,
            length_penalty=2.0,
            early_stopping=False,
            min_length=min_length,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for item, response in zip(batch_items, responses):
            translation = response.split("Output:")[-1].strip()
            translation = translation.split("\n")[0].strip()
            translation = translation.split("Translate this")[
                0
            ].strip()  # remove any prompt text
            translation = translation.split("Translation:")[
                0
            ].strip()  # remove any extra translation markers
            translation = translation.replace("\n", " ")
            translation = translation.strip("\"'")

            # calculate expected minimum length based on language pair expansion factor
            expansion_factor = get_expansion_factor(
                item["source_language"], item["target_language"]
            )
            source_tokens = len(tokenizer.encode(item["src"]))
            expected_min_tokens = int(source_tokens * expansion_factor * 0.8)

            translation_tokens = len(tokenizer.encode(translation))
            if translation_tokens < min(20, expected_min_tokens):
                logging.warning(
                    f"Translation might be incomplete for item {item['id']}. "
                    f"Source tokens: {source_tokens}, "
                    f"Translation tokens: {translation_tokens}, "
                    f"Expected min tokens: {expected_min_tokens}, "
                    f"Expansion factor: {expansion_factor:.2f}"
                )

            output_item = item.copy()
            output_item["mt"] = translation
            all_translations.append(output_item)

    return all_translations


def translate_batch_with_timing(
    items: List[Dict],
) -> Tuple[List[Dict], float]:

    start_time = time.time()
    translations = translate_batch(items)
    end_time = time.time()

    return translations, end_time - start_time
