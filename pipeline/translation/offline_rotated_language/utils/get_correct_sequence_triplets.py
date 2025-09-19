from typing import List, Dict
from pipeline.translation.offline_rotated_language.utils.set_seq_from_lang_pair import (
    set_seq_from_lang_pair,
)


def get_correct_sequence_triplets(
    items: List[Dict], triplets_dict: Dict, iteration: int, is_direct: bool = False
) -> List[Dict]:
    for item in items:
        sequence = set_seq_from_lang_pair(
            source_language=item["original_source_language"],
            target_language=item["original_target_language"],
            triplets_dict=triplets_dict,
            is_direct=is_direct,
        )
        current_language = sequence[((iteration - 2) + len(sequence)) % len(sequence)]
        target_language = sequence[((iteration - 1) + len(sequence)) % len(sequence)]
        item["source_language"] = current_language
        item["target_language"] = target_language
        item["src"] = item["mt"]

    return items
