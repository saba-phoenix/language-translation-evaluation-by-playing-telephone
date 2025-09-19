import os
import sys
from typing import List, Dict, Any

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("wordnet")
except LookupError:
    nltk.download("wordnet")


def evaluate_meteor_single(text_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Evaluates METEOR score on a list of text paragraphs.

    Args:
        text_list: List of dictionaries containing text paragraphs with 'mt' and 'ref' keys

    Returns:
        List of dictionaries containing paragraph data with METEOR scores
    """
    eval_list = []

    for paragraph in text_list:
        mt = paragraph["mt"]
        ref = paragraph["ref"]

        mt_tokens = word_tokenize(mt.lower())
        ref_tokens = word_tokenize(ref.lower())

        meteor_score_value = meteor_score([ref_tokens], mt_tokens)

        eval_list.append({**paragraph, "meteor_score": meteor_score_value})

    return eval_list


def evaluate(text_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Evaluates METEOR scores for the provided list of texts.

    Parameters:
        - text_list: A list of text paragraphs to evaluate. Each paragraph should contain
                    'mt' (machine translation) and 'ref' (reference translation) keys.

    Returns:
        List of dictionaries containing paragraph data with METEOR scores.
    """
    return evaluate_meteor_single(text_list=text_list)
