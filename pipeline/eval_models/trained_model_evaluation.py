import os
import sys
from typing import List, Dict, Any

import torch

torch.set_float32_matmul_precision("high")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comet import download_model, load_from_checkpoint

from pipeline.utils.data_utils.constants import XCOMET_MODEL


# lazy initialization of the COMET model
_comet_model = None

# put the model name here
MODEL_NAME = "era28/unifiedmetrics-model-rotation-hmref-ftmqm"


def get_comet_model():
    """
    Lazy-loads and returns the COMET model.
    The model is downloaded and loaded only once.
    """
    global _comet_model
    if _comet_model is None:
        model_path = download_model(MODEL_NAME)
        _comet_model = load_from_checkpoint(model_path)
    return _comet_model


def cleanup_comet_model():
    """
    Cleans up the COMET model instance and frees associated resources.
    """
    global _comet_model
    if _comet_model is not None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _comet_model = None


def evaluate_comet_single(
    text_list: List[Dict[str, Any]],
    comet_model: Any,
    batch_size: int = 8,
    gpus: int = 1,
) -> List[Dict[str, Any]]:
    """
    Evaluates the COMET model on a list of paragraphs.

    Args:
        text_list: List of dictionaries containing text paragraphs
        comet_model: The COMET model instance
        batch_size: Batch size for model prediction
        gpus: Number of GPUs to use

    Returns:
        List of dictionaries containing paragraph data with xcomet scores
    """
    model_output = comet_model.predict(text_list, batch_size=batch_size, gpus=gpus)
    scores = model_output["scores"]
    # Ensure scores are between 0 and 1
    scores = torch.clamp(torch.tensor(scores), min=0.0, max=1.0).tolist()

    return [
        {**paragraph, "xcomet_score": score}
        for paragraph, score in zip(text_list, scores)
    ]


def evaluate(text_list, batch_size=8, gpus=1):
    """
    Downloads (if needed) and loads the COMET model via lazy initialization,
    then evaluates the provided list of texts. Cleans up the model after use.

    Parameters:
        - text_list: A list of text paragraphs to evaluate.

    Returns:
        A tuple containing the average COMET score and detailed error spans.
    """
    try:
        comet_model = get_comet_model()
        return evaluate_comet_single(
            text_list=text_list,
            comet_model=comet_model,
            batch_size=batch_size,
            gpus=gpus,
        )
    finally:
        cleanup_comet_model()
