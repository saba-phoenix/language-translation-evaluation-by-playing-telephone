import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_babel_model = None
_babel_tokenizer = None
_babel_model_initialized = False


_DEVICE = "cuda"
_MODEL_NAME = "Tower-Babel/Babel-9B-Chat"


def get_model():

    global _babel_model, _babel_tokenizer, _babel_model_initialized

    if not _babel_model_initialized:
        print(f"Loading Tower-Babel model '{_MODEL_NAME}' onto '{_DEVICE}'...")
        _babel_model = AutoModelForCausalLM.from_pretrained(
            _MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map=_DEVICE,
        )
        _babel_tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        if _babel_tokenizer.pad_token is None:
            _babel_tokenizer.pad_token = _babel_tokenizer.eos_token

        _babel_model_initialized = True
        print("Model loaded.")
    return _babel_tokenizer, _babel_model


def translate(
    text,
    target_language,
    source_language,
    max_length=512,
):
    tokenizer, model = get_model()

    messages = [
        {
            "role": "system",
            "content": f"You are a helpful assistant that translates text from {source_language} into {target_language}.",
        },
        {
            "role": "user",
            "content": text,
        },
    ]
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    model_inputs = tokenizer(
        [prompt_str],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    model_inputs = {k: v.to(_DEVICE) for k, v in model_inputs.items()}

    generated_ids = model.generate(
        model_inputs["input_ids"],
        attention_mask=model_inputs["attention_mask"],
        max_new_tokens=max_length,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs["input_ids"], generated_ids)
    ]

    translated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return translated_text
