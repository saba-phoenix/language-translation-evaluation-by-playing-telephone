import os
import sys
from typing import List, Dict, Any

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sacrebleu


def evaluate_bleu_single(text_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Evaluates BLEU score on a list of text paragraphs.

    Args:
        text_list: List of dictionaries containing text paragraphs with 'mt' and 'ref' keys

    Returns:
        List of dictionaries containing paragraph data with BLEU scores
    """
    eval_list = []

    for paragraph in text_list:
        mt = paragraph["mt"]
        ref = paragraph["ref"]

        # Calculate BLEU score
        bleu = sacrebleu.sentence_bleu(mt, [ref])
        bleu_score = bleu.score

        eval_list.append({**paragraph, "bleu_score": bleu_score})

    return eval_list


def evaluate(text_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Evaluates BLEU scores for the provided list of texts.

    Parameters:
        - text_list: A list of text paragraphs to evaluate. Each paragraph should contain
                    'mt' (machine translation) and 'ref' (reference translation) keys.

    Returns:
        List of dictionaries containing paragraph data with BLEU scores.
    """
    return evaluate_bleu_single(text_list=text_list)
