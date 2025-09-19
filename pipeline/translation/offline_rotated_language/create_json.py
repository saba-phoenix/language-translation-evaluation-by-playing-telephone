import pandas as pd
import json
import os


def create_offline_json():
    df = pd.read_csv("data/unique_da.csv")

    json_data = []
    for idx, row in df.iterrows():
        entry = {
            "id": f"{idx}_{row['tgt_lang']}_{row['src_lang']}_0",
            "src": row["ref"],
            "mt": row["src"],
            "source_language": row["tgt_lang"],
            "target_language": row["src_lang"],
        }
        json_data.append(entry)

    output_dir = "outputs/batch_output"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "offline_0.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print(f"Created {output_file} with {len(json_data)} entries")


if __name__ == "__main__":
    create_offline_json()
