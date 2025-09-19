import gc
import traceback
from pipeline.translation.offline.create_single_ip_op_with_bt import (
    create_full_ip_op_aggregate,
)
from pipeline.translation.offline.evaluation import run_eval_pipeline
from pipeline.translation.offline_rotated_translation_d import (
    main as rotated_translation_main,
)


def main():
    output_file_name = "ip_op_mr.json"
    input_file_prefix = "offline_"
    input_file_suffix = "_mr"
    num_iterations = 18
    batch_size = 128

    try:
        create_full_ip_op_aggregate(
            output_file_name=output_file_name,
            num_iterations=num_iterations,
            input_file_prefix=input_file_prefix,
            input_file_suffix=input_file_suffix,
        )

        run_eval_pipeline(
            input_file_name=output_file_name,
            output_file_name=output_file_name,
            iteration_num=num_iterations,
            batch_size=batch_size,
            include_ref=False,
        )

    except Exception as e:
        print("Error occurred during preprocessing or evaluation.")
        traceback.print_exc()  # More detailed error output

    finally:
        gc.collect()
        print("Starting rotated translation")
        print("-" * 100)
        rotated_translation_main()


if __name__ == "__main__":
    main()
