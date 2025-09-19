# import openai
from config import KEY
from openai import OpenAI

_client_initialized = False
client = None


def get_client():
    """
    Returns an initialized openai client, setting up API key on the first call only.
    """
    global _client_initialized
    global client

    if not _client_initialized:
        # Set the API key or any other configuration here
        # openai.api_key = KEY
        client = OpenAI(api_key=KEY)

        # In principle, you could create a custom client object if needed,
        # but the openai library uses a global state pattern for `api_key`.
        _client_initialized = True

    # Return the openai module (with the key set) as our 'client'.
    return client


def generate_response(messages, model, temperature=0.2):
    """
    Dynamically include `temperature` param if the model supports it.
    """

    client = get_client()  # Acquire the lazy-loaded client
    # Known set of models/endpoints that allow temperature
    models_with_temperature = {"gpt-3.5-turbo", "gpt-4", "gpt-4o" "text-davinci-003"}

    params = {
        "model": model,
        "messages": messages,
        # Possibly other shared params
    }

    # Add temperature if we are using a model that supports it
    if model in models_with_temperature and temperature is not None:
        params["temperature"] = temperature

    # Now create the completion
    response = client.chat.completions.create(**params)

    return response.choices[0].message.content


def prepare_translation(text, temperature, target_language, source_language, model):
    """
    Constructs a specialized translation command and calls generate_response.
    """
    command = (
        f"Translate the text into {target_language}, ensuring that programming terms and book terms "
        f"are transliterated using the {target_language} script while preserving their English"
        f"pronunciation. Do not translate the code blocks, leave them as they are. "
        f"Mathematical variable names should remain in English letters, but all other "
        f"content must be fully translated without using {source_language} words. Ensure the "
        f"translation is engaging and natural, preserving the original tone and style. "
        f"Don't keep {source_language} words, translate them. Do not remove any å, @ and []. "
        f"Do not erase or include any escape characters and all non alphanumeric characters. "
        f"Retain all square brackets at any cost. The text to be translated is as follows:"
    )

    messages = [
        {
            "role": "system",
            "content": f"You are an {source_language} to {target_language} Translator.",
        },
        {"role": "user", "content": f"{command} {text}"},
    ]

    return generate_response(messages=messages, model=model, temperature=temperature)


def translate(text, target_language, source_language, model):
    """
    Public-facing function to translate text to the specified language.
    """
    # Example: using a moderate temperature for balanced translations
    return prepare_translation(
        text=text,
        temperature=0.20,
        target_language=target_language,
        source_language=source_language,
        model=model,
    )
