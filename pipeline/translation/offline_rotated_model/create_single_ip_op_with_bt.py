import os
import json


def create_full_ip_op_aggregate(
    output_file_name=None,
    num_iterations=18,
    input_file_prefix="offline_",
    input_file_suffix="_200",
):
    input_dir = "outputs/batch_output"
    combined_dict = {}

    offline_0_path = os.path.join(input_dir, f"{input_file_prefix}0.json")
    with open(offline_0_path, "r", encoding="utf-8") as f:
        lang_data = json.load(f)
        lang_map = {
            item["id"].split("_")[0]: {
                "source_language": item["target_language"],  # Flip
                "target_language": item["source_language"],
            }
            for item in lang_data
        }

    # Iterate through offline files in pairs (0-1, 2-3, ..., 16-17)
    for i in range(0, num_iterations, 2):
        if i == 0:
            input_file = os.path.join(input_dir, f"{input_file_prefix}{i}.json")

        else:
            input_file = os.path.join(
                input_dir, f"{input_file_prefix}{i}{input_file_suffix}.json"
            )
        output_file = os.path.join(
            input_dir, f"{input_file_prefix}{i+1}{input_file_suffix}.json"
        )

        if not os.path.exists(input_file) or not os.path.exists(output_file):
            print(f"Skipping pair {i}-{i+1}: missing file")
            continue

        with open(input_file, "r", encoding="utf-8") as f_in, open(
            output_file, "r", encoding="utf-8"
        ) as f_out:
            input_data = json.load(f_in)
            output_data = json.load(f_out)

            input_data = sorted(input_data, key=lambda x: int(x["id"].split("_")[0]))
            output_data = sorted(output_data, key=lambda x: int(x["id"].split("_")[0]))

            for inp, out in zip(input_data, output_data):
                idx = inp["id"].split("_")[0]
                idx_op = out["id"].split("_")[0]

                if idx != idx_op:
                    print(f"id: {inp['id']}")
                    print(f"id_op: {out['id']}")
                    print(f"idx: {idx}, idx_op: {idx_op}")
                    raise ValueError("idx and idx_op are not the same")

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
                if i == 0:
                    combined_dict[idx][f"ref"] = inp["src"]
                combined_dict[idx][f"input_{iteration_num}"] = inp["mt"]
                combined_dict[idx][f"output_{iteration_num}"] = out["mt"]

    final_list = list(combined_dict.values())

    output_path = os.path.join(input_dir, output_file_name)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_list, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(final_list)} records to {output_path}")


if __name__ == "__main__":
    create_full_ip_op_aggregate(
        output_file_name="ip_op_mr_ref.json",
        num_iterations=18,
        input_file_prefix="offline_",
        input_file_suffix="_mrc",
    )
