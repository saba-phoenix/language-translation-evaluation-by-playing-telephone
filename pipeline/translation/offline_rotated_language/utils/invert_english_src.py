from typing import List, Dict


def invert_english_src(items: List[Dict]) -> List[Dict]:
    for item in items:
        if "original_source_language" not in item.keys():
            if item["target_language"] == "English":
                # "id": "0_Czech_English_0",
                id, source_language, target_language, iteration = item["id"].split("_")
                item["id"] = f"{id}_{target_language}_{source_language}_{iteration}"
                item["src"], item["mt"] = item["mt"], item["src"]
                item["source_language"] = target_language
                item["target_language"] = source_language
                item["original_source_language"] = target_language
                item["original_target_language"] = source_language
            else:
                item["original_source_language"] = item["source_language"]
                item["original_target_language"] = item["target_language"]
    return items
