import os
import sys
from typing import List, Dict, Any

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sacrebleu


def evaluate_ter_single(text_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Evaluates TER (Translation Edit Rate) score on a list of text paragraphs.
    Note: TER scores are negated so that higher values indicate better translations.

    Args:
        text_list: List of dictionaries containing text paragraphs with 'mt' and 'ref' keys

    Returns:
        List of dictionaries containing paragraph data with negated TER scores
    """
    eval_list = []

    for paragraph in text_list:
        mt = paragraph["mt"]
        ref = paragraph["ref"]

        # Calculate TER score
        ter = sacrebleu.sentence_ter(mt, [ref])
        ter_score = ter.score

        # Negate the TER score so that higher values indicate better translations
        negated_ter_score = -ter_score

        eval_list.append({**paragraph, "ter_score": negated_ter_score})

    return eval_list


def evaluate(text_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Evaluates negated TER scores for the provided list of texts.

    Parameters:
        - text_list: A list of text paragraphs to evaluate. Each paragraph should contain
                    'mt' (machine translation) and 'ref' (reference translation) keys.

    Returns:
        List of dictionaries containing paragraph data with negated TER scores.
        Note: Higher values now indicate better translations.
    """
    return evaluate_ter_single(text_list=text_list)
