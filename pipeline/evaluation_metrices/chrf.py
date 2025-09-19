import os
import sys
from typing import List, Dict, Any

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sacrebleu


def evaluate_chrf_single(text_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Evaluates CHRF score on a list of text paragraphs.

    Args:
        text_list: List of dictionaries containing text paragraphs with 'mt' and 'ref' keys

    Returns:
        List of dictionaries containing paragraph data with CHRF scores
    """
    eval_list = []

    for paragraph in text_list:
        mt = paragraph["mt"]
        ref = paragraph["ref"]

        # Calculate CHRF score
        chrf = sacrebleu.sentence_chrf(mt, [ref])
        chrf_score = chrf.score

        eval_list.append({**paragraph, "chrf_score": chrf_score})

    return eval_list


def evaluate(text_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Evaluates CHRF scores for the provided list of texts.

    Parameters:
        - text_list: A list of text paragraphs to evaluate. Each paragraph should contain
                    'mt' (machine translation) and 'ref' (reference translation) keys.

    Returns:
        List of dictionaries containing paragraph data with CHRF scores.
    """
    return evaluate_chrf_single(text_list=text_list)
