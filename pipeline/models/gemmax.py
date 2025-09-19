import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_gemmax_model = None
_gemmax_tokenizer = None
_model_initialized = False

_MODEL_ID = "ModelSpace/GemmaX2-28-2B-v0.1"


def get_model():
    global _gemmax_model, _gemmax_tokenizer, _model_initialized

    if not _model_initialized:
        _gemmax_tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
        _gemmax_model = AutoModelForCausalLM.from_pretrained(_MODEL_ID)
        _model_initialized = True

    return _gemmax_model, _gemmax_tokenizer


def translate(
    text: str,
    source_language: str = "English",
    target_language: str = "Bangla",
) -> str:
    """
    Translates the given text from `source_language` to `target_language`.
    Adjust the prompt or parameters as needed.
    """
    model, tokenizer = get_model()

    prompt = (
        f"Translate this from {source_language} to {target_language}:\n"
        f"{source_language}: {text}\n"
        f"{target_language}:"
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=len(text.split()) * 5)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    translation_part = response.split(f"{target_language}:")[-1]
    # print(translation_part)
    return translation_part
