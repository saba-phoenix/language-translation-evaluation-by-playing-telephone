import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

_gpt2_model = None
_gpt2_tokenizer = None
_gpt2_model_initialized = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_model():
    global _gpt2_model, _gpt2_tokenizer, _gpt2_model_initialized

    if not _gpt2_model_initialized:
        print("Loading GPT-2 model...")
        _gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        _gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)

        _gpt2_model_initialized = True
        print("GPT-2 Model Loaded.")

    return _gpt2_tokenizer, _gpt2_model


def translate(text, source_language="English", target_language="French"):

    tokenizer, model = get_model()

    prompt = (
        f"Translate the following {source_language} text to {target_language}:\n"
        f"{source_language}: '{text}'\n"
        f"{target_language}:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    output = model.generate(
        **inputs,
        max_length=1024,
        num_return_sequences=1,
        temperature=0.2,
    )

    translation = tokenizer.decode(output[0], skip_special_tokens=True)

    translation = translation.replace(prompt, "").strip()

    return translation
