import os
import json


def create_full_ip_op_aggregate(output_file_name=None, num_iterations=18):
    output_dir = "outputs/batch_output"
    combined_dict = {}

    offline_0_path = os.path.join(output_dir, "offline_0.json")
    with open(offline_0_path, "r", encoding="utf-8") as f:
        lang_data = json.load(f)
        lang_map = {
            item["id"].split("_")[0]: {
                "source_language": item["target_language"],  # Flip
                "target_language": item["source_language"],
            }
            for item in lang_data
        }

    # iterate through offline files in pairs (0-1, 2-3, ..., 16-17)
    for i in range(0, num_iterations, 2):
        input_file = os.path.join(output_dir, f"offline_{i}.json")
        output_file = os.path.join(output_dir, f"offline_{i+1}.json")

        if not os.path.exists(input_file) or not os.path.exists(output_file):
            print(f"Skipping pair {i}-{i+1}: missing file")
            continue

        with open(input_file, "r", encoding="utf-8") as f_in, open(
            output_file, "r", encoding="utf-8"
        ) as f_out:
            input_data = json.load(f_in)
            output_data = json.load(f_out)

            for inp, out in zip(input_data, output_data):
                idx = inp["id"].split("_")[0]
                iteration_num = (i // 2) + 1

                if idx not in combined_dict:
                    combined_dict[idx] = {
                        "id": idx,
                        **lang_map.get(
                            idx,
                            {
                                "source_language": inp[
                                    "target_language"
                                ],  # fallback if somehow missing
                                "target_language": inp["source_language"],
                            },
                        ),
                    }

                combined_dict[idx][f"input_{iteration_num}"] = inp["mt"]
                combined_dict[idx][f"output_{iteration_num}"] = out["mt"]

    final_list = list(combined_dict.values())
    output_path = os.path.join(output_dir, output_file_name)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_list, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(final_list)} records to {output_path}")


if __name__ == "__main__":
    create_full_ip_op_aggregate("ip_op.json", num_iterations=18)
