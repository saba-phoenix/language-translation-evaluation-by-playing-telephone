import os
import json


def create_full_ip_op_aggregate(
    output_file_name=None,
    num_iterations=18,
    input_file_prefix="offline_",
    input_file_suffix="_mr",
):
    input_dir = "outputs/batch_output"
    output_dir = "outputs/all_output"
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

    for i in range(1, num_iterations + 1):
        if i in [3, 16, 17, 18]:
            continue
        # elif i == 0:
        #     input_file = os.path.join(input_dir, f"{input_file_prefix}{i}.json")
        else:
            input_file = os.path.join(
                input_dir, f"{input_file_prefix}{i}{input_file_suffix}.json"
            )

        if not os.path.exists(input_file):
            print(f"Skipping file {i}: missing file")
            continue

        with open(input_file, "r", encoding="utf-8") as f_in:
            input_data = json.load(f_in)

            for inp in input_data:
                idx = inp["id"].split("_")[0]
                iteration_num = i

                if idx not in combined_dict:
                    combined_dict[idx] = {
                        "id": idx,
                        **lang_map.get(
                            idx,
                            {
                                "source_language": inp["target_language"],
                                "target_language": inp["source_language"],
                            },
                        ),
                    }
                if i == 1:
                    combined_dict[idx][f"input_1"] = inp["src"]
                # combined_dict[idx][f"input_{iteration_num}"] = inp["mt"]
                combined_dict[idx][f"output_{iteration_num}"] = inp["mt"]

    final_list = list(combined_dict.values())

    output_path = os.path.join(output_dir, output_file_name)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_list, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(final_list)} records to {output_path}")


if __name__ == "__main__":
    create_full_ip_op_aggregate(
        output_file_name="ip_op_lowdiv.json",
        num_iterations=18,
        input_file_prefix="offline_",
        input_file_suffix="_lowdiv",
    )
