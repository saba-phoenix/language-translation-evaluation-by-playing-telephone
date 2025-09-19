import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import sys
import pandas as pd

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comet import download_model, load_from_checkpoint

from pipeline.utils.data_utils.constants import COMET_MODEL


# Global for lazy initialization of the COMET model
_comet_model = None


def get_comet_model():
    """
    Lazy-loads and returns the COMET model.
    The model is downloaded and loaded only once.
    """
    global _comet_model
    if _comet_model is None:
        # Download the model (if not already cached)
        model_path = download_model(COMET_MODEL)
        # Load the model from the checkpoint
        _comet_model = load_from_checkpoint(model_path)
    return _comet_model


def evaluate_comet_single(text_list, comet_model):
    """
    Evaluates the COMET model on a list of paragraphs.

    Parameters:
        - comet_model: The COMET model instance.

    Returns:
        A tuple containing the average score and a list of error span details.
    """
    # Evaluate the model
    model_output = comet_model.predict(text_list, batch_size=8, gpus=1)
    eval_list = []
    scores = model_output["scores"]
    for paragraph, score in zip(text_list, scores):
        eval_list.append(
            {
                **paragraph,
                "id": paragraph["id"],
                "src": paragraph["src"],
                "mt": paragraph["mt"],
                "comet_score": score,
            }
        )
    # avg_score = sum(scores) / len(scores)
    return eval_list


def evaluate(text_list):
    """
    Downloads (if needed) and loads the COMET model via lazy initialization,
    then evaluates the provided list of texts.

    Parameters:
        - text_list: A list of text paragraphs to evaluate.

    Returns:
        A tuple containing the average COMET score and detailed eval.
    """
    comet_model = get_comet_model()
    return evaluate_comet_single(text_list=text_list, comet_model=comet_model)
