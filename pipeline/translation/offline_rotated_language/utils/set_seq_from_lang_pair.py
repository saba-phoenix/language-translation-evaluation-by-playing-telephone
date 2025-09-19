from typing import Dict, List


def set_seq_from_lang_pair(
    source_language: str, target_language: str, triplets_dict: Dict, is_direct=False
) -> List[str]:
    triplets = triplets_dict[target_language]
    if is_direct:
        return [triplets[1], triplets[2], triplets[0]]
    else:
        return ["English", triplets[1], "English", triplets[2], "English", triplets[0]]
