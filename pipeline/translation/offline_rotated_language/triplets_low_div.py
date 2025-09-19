import os
import gc
import logging
import random
from datetime import datetime
import signal
import sys
import psutil
import traceback
from logging.handlers import RotatingFileHandler

from tqdm import tqdm
import json

from typing import List, Dict
from pipeline.models.gemma3_12b_batch import (
    translate_batch_with_timing as gemma_translate,
)
from pipeline.utils.data_utils.rotated_lang import low_div_triplets
from pipeline.translation.offline_rotated_language.utils.set_seq_from_lang_pair import (
    set_seq_from_lang_pair,
)
from pipeline.translation.offline_rotated_translation_d import (
    main as rotated_translation_main,
)
from pipeline.translation.offline_rotated_language.utils.invert_english_src import (
    invert_english_src,
)
from pipeline.translation.offline_rotated_language.utils.get_correct_sequence_triplets import (
    get_correct_sequence_triplets,
)
from pipeline.translation.offline_rotated_language.triplets_high_div import (
    main as high_div_main,
)
from pipeline.translation.offline_rotated_language.triplets_low_div_dir import (
    main as low_div_dir_main,
)
from pipeline.translation.offline_rotated_language.triplets_high_div_dir import (
    main as high_div_dir_main,
)


def setup_logging():
    """Setup logging configuration with separate error log file."""
    log_dir = "outputs/logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Main log file
    log_file = os.path.join(log_dir, f"translation_{timestamp}.log")

    # Error log file
    error_log_file = os.path.join(log_dir, f"translation_errors_{timestamp}.log")

    # Configure main logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    # Configure error logger
    error_logger = logging.getLogger("error_logger")
    error_logger.setLevel(logging.ERROR)
    error_handler = RotatingFileHandler(
        error_log_file, maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
    )
    error_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(levelname)s - %(processName)s - %(process)d - %(message)s"
        )
    )
    error_logger.addHandler(error_handler)

    return log_file, error_log_file


def log_process_info():
    """Log current process information."""
    process = psutil.Process()
    return {
        "pid": process.pid,
        "name": process.name(),
        "username": process.username(),
        "create_time": datetime.fromtimestamp(process.create_time()).strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "memory_info": process.memory_info()._asdict(),
        "cpu_percent": process.cpu_percent(),
    }


def log_error(error_logger, error, context=None):
    """Log error with process information and context."""
    process_info = log_process_info()
    error_msg = {
        "error": str(error),
        "process_info": process_info,
        "context": context,
        "stack_trace": traceback.format_exc(),
    }
    error_logger.error(json.dumps(error_msg, indent=2))


def save_iteration(
    items: List[Dict], iteration: int, is_intermediate: bool = False, batch_num: int = 0
):
    """Save the current iteration results to a JSON file."""
    output_dir = "outputs/batch_output"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"offline_{iteration}_lowdiv1.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    logging.info(
        f"Saved {'intermediate ' if is_intermediate else ''}iteration {iteration} to {output_file}"
    )


def main():
    try:
        max_iterations = 18
        current_iteration = 10
        # Setup logging and signal handlers
        log_file, error_log_file = setup_logging()
        error_logger = logging.getLogger("error_logger")

        logging.info(f"Starting translation process. Log file: {log_file}")
        logging.info(f"Error log file: {error_log_file}")
        logging.info(f"Process Info: {json.dumps(log_process_info(), indent=2)}")

        # Load initial data
        with open(
            f"outputs/batch_output/offline_{current_iteration-1}_lowdiv1.json",
            "r",
            encoding="utf-8",
        ) as f:
            items = json.load(f)

        # items = random.sample(items, 100)

        items = invert_english_src(items)
        ## sample 100
        triplets_dict = low_div_triplets

        # Define the sequence of models to use
        model_group = (gemma_translate, "Gemma3", 24)

        ## group the items by their source and target languages, keep all the items with the same source and target languages together
        items_by_lang = {}
        for item in items:
            key = (item["source_language"], item["target_language"])
            if key not in items_by_lang:
                items_by_lang[key] = []
            items_by_lang[key].append(item)

        # for each of the id based on their source and target language, set a sequence

        for iteration in range(current_iteration, max_iterations + 1):
            try:
                logging.info(f"\nStarting iteration {iteration}")

                items = get_correct_sequence_triplets(items, triplets_dict, iteration)

                translate_func, model_name, model_batch_size = model_group
                logging.info(f"Using {model_name} for translation")

                batch_size = 35000
                all_translated_items = []

                for batch_start in range(0, len(items), batch_size):
                    try:
                        # if (batch_start//batch_size + 1) < 7:
                        #     continue
                        batch_end = min(batch_start + batch_size, len(items))
                        batch_items = items[batch_start:batch_end]
                        logging.info(
                            f"Processing batch {batch_start//batch_size + 1} of {(len(items) + batch_size - 1)//batch_size}"
                        )

                        translated_batch, processing_time = translate_func(
                            batch_items,
                            model_batch_size,
                            clear_cache=True,
                            is_last=False,
                        )
                        logging.info(
                            f"Batch translation completed in {processing_time:.2f} seconds"
                        )

                        all_translated_items.extend(translated_batch)
                        save_iteration(
                            all_translated_items,
                            iteration,
                            is_intermediate=True,
                            batch_num=batch_start // batch_size + 1,
                        )
                    except Exception as e:
                        log_error(
                            error_logger,
                            e,
                            {
                                "iteration": iteration,
                                "batch_start": batch_start,
                                "batch_size": batch_size,
                                "model_name": model_name,
                            },
                        )

                save_iteration(all_translated_items, iteration)
                items = all_translated_items

            except Exception as e:
                log_error(
                    error_logger, e, {"iteration": iteration, "model_name": model_name}
                )

    except Exception as e:
        log_error(error_logger, e, {"context": "main process"})


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Fallback if main() fails
        fallback_msg = {
            "error": str(e),
            "stack_trace": traceback.format_exc(),
        }
        sys.stderr.write("ERROR in main()\n")
        sys.stderr.write(json.dumps(fallback_msg, indent=2))
        sys.stderr.write("\n")
    finally:
        gc.collect()
        # high_div_main()
        sys.stderr.write("Running rotated_translation_main()...\n")
        rotated_translation_main()
