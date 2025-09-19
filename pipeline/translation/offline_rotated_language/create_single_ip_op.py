import os
import json


def create_full_ip_op_aggregate(op_file_name=None, num_iterations=18):
    output_dir = "outputs/batch_output"
    combined_dict = {}

    for i in range(0, num_iterations + 1):
        input_file = os.path.join(output_dir, f"offline_{i}_50.json")

        if not os.path.exists(input_file):
            print(f"Skipping iteration {i}: missing file")
            continue

        with open(input_file, "r", encoding="utf-8") as f_in:
            input_data = json.load(f_in)

            for inp in input_data:
                idx = inp["id"].split("_")[0]
                iteration_num = i

                if idx not in combined_dict:
                    combined_dict[idx] = {
                        "id": idx,
                    }
                if iteration_num == 0:
                    combined_dict[idx][f"input_1"] = inp["mt"]
                else:
                    combined_dict[idx][f"output_{iteration_num}"] = inp["mt"]

    final_list = list(combined_dict.values())

    output_path = os.path.join(output_dir, op_file_name)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_list, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(final_list)} records to {output_path}")


if __name__ == "__main__":
    create_full_ip_op_aggregate(op_file_name="ip_op_lang_50.json", num_iterations=18)
