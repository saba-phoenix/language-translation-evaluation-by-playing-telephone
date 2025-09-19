import os
import logging
from datetime import datetime
import signal
import sys
import psutil
import traceback
from logging.handlers import RotatingFileHandler

# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from tqdm import tqdm
import json

from typing import List, Dict
from pipeline.models.gemma3_12b_batch_old import (
    translate_batch_with_timing as gemma_translate,
)
from pipeline.models.suzume_batch import translate_batch_with_timing as suzume_translate
from pipeline.models.nllb_batch_old import translate_batch_with_timing as nllb_translate


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


def signal_handler(signum, frame):
    """Handle termination signals."""
    error_logger = logging.getLogger("error_logger")
    process_info = log_process_info()
    error_logger.error(
        f"Process terminated by signal {signum}\n"
        f"Process Info: {json.dumps(process_info, indent=2)}\n"
        f"Stack trace: {traceback.format_stack()}"
    )
    sys.exit(0)


def setup_signal_handlers():
    """Setup handlers for termination signals."""
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


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


def flip_languages(items: List[Dict]) -> List[Dict]:
    """Flip source and target languages for each item."""
    for item in items:
        whole_id = item["id"]
        idx, src_id, tgt_id, it = (
            whole_id.split("_")[0],
            whole_id.split("_")[1],
            whole_id.split("_")[2],
            int(whole_id.split("_")[-1]),
        )
        item["id"] = f"{idx}_{tgt_id}_{src_id}_{it+1}"
        item["source_language"], item["target_language"] = (
            item["target_language"],
            item["source_language"],
        )
        item["src"], item["mt"] = item["mt"], item["src"]
    return items


def save_iteration(
    items: List[Dict], iteration: int, is_intermediate: bool = False, batch_num: int = 0
):
    """Save the current iteration results to a JSON file."""
    output_dir = "outputs/batch_output"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"offline_{iteration}.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    logging.info(
        f"Saved {'intermediate ' if is_intermediate else ''}iteration {iteration} to {output_file}"
    )


def main():
    try:
        # Setup logging and signal handlers
        log_file, error_log_file = setup_logging()
        setup_signal_handlers()
        error_logger = logging.getLogger("error_logger")

        logging.info(f"Starting translation process. Log file: {log_file}")
        logging.info(f"Error log file: {error_log_file}")
        logging.info(f"Process Info: {json.dumps(log_process_info(), indent=2)}")

        # Load initial data
        with open("outputs/batch_output/offline_7.json", "r", encoding="utf-8") as f:
            items = json.load(f)

        # Define the sequence of models to use
        model_sequence = [
            (gemma_translate, "Gemma3", 16),
            (gemma_translate, "Gemma3", 16),
            (nllb_translate, "NLLB", 16),
            (nllb_translate, "NLLB", 16),
            (suzume_translate, "Suzume", 16),
            (suzume_translate, "Suzume", 16),
        ]

        for iteration in range(8, 19):
            try:
                logging.info(f"\nStarting iteration {iteration}")

                items = flip_languages(items)
                model_idx = (iteration - 1) % len(model_sequence)
                translate_func, model_name, model_batch_size = model_sequence[model_idx]
                logging.info(f"Using {model_name} for translation")

                batch_size = 10000
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
                            batch_items, model_batch_size
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
    main()
