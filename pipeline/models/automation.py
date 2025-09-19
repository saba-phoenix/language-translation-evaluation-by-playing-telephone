import json
import os
import time
import logging
from datetime import datetime
from openai import OpenAI
from typing import List, Dict, Any
from config import KEY, TOKEN, BATCH_FILE_ID, BATCH_ID

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("batch_pipeline.log"), logging.StreamHandler()],
)
logger = logging.getLogger("batch_pipeline")


# Configuration
class Config:
    API_KEY = KEY
    COMPLETION_WINDOW = "24h"
    OUTPUT_DIR = "outputs"
    BATCH_REQUEST_DIR = f"{OUTPUT_DIR}/batch_request"
    BATCH_OUTPUT_DIR = f"{OUTPUT_DIR}/batch_output"
    BATCH_METRICS_DIR = f"{OUTPUT_DIR}/batch_metrics"

    MODELS = {
        1: "gpt-4.1",
        2: "gpt-4.1",
        3: "gpt-4.1-mini",
        4: "gpt-4.1-mini",
        5: "gpt-4o-mini",
        6: "gpt-4o-mini",
    }

    # determine which model to use based on iteration number
    @classmethod
    def get_model_for_iteration(cls, iteration: int) -> str:
        model_idx = ((iteration - 1) % 6) + 1
        return cls.MODELS[model_idx]

    # determine if this is a back translation based on iteration number
    @classmethod
    def is_back_translation(cls, iteration: int) -> bool:
        return iteration % 2 == 0


def create_directories():
    os.makedirs(Config.BATCH_REQUEST_DIR, exist_ok=True)
    os.makedirs(Config.BATCH_OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.BATCH_METRICS_DIR, exist_ok=True)


def get_client():
    return OpenAI(api_key=Config.API_KEY)


def build_batch_item(
    idx: int,
    text: str,
    source_language: str,
    target_language: str,
    model: str,
    temperature: float = 0.2,
    iteration: int = 1,
) -> Dict[str, Any]:

    command = f"Translate the following text into {target_language}."

    messages = [
        {
            "role": "system",
            "content": f"You are an {source_language} to {target_language} Translator.",
        },
        {"role": "user", "content": f"{command}\n\n{text}"},
    ]

    return {
        "custom_id": f"{source_language}_to_{target_language}_{idx}_{iteration}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"model": model, "messages": messages, "temperature": temperature},
    }


def write_jsonl_batch(
    translation_jobs: List[Dict[str, Any]],
    model: str,
    iteration: int,
    temperature: float = 0.2,
) -> str:

    output_file = f"{Config.BATCH_REQUEST_DIR}/it_{iteration}.jsonl"
    logger.info(f"Writing batch request file: {output_file}")

    with open(output_file, "w", encoding="utf-8") as f:
        for job in translation_jobs:
            item = build_batch_item(
                idx=job["id"],
                text=job["text"],
                source_language=job["source_language"],
                target_language=job["target_language"],
                model=model,
                temperature=temperature,
                iteration=iteration,
            )
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return output_file


def create_batch(file_path: str, iteration: int) -> str:

    client = get_client()

    with open(file_path, "rb") as f:
        batch_input_file = client.files.create(file=f, purpose="batch")

    logger.info(f"Uploaded file with ID: {batch_input_file.id}")

    batch_response = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window=Config.COMPLETION_WINDOW,
        metadata={"description": f"Iteration {iteration} translation batch"},
    )

    batch_id = batch_response.id
    logger.info(f"Created batch with ID: {batch_id} for iteration {iteration}")

    return batch_id


def monitor_batch(batch_id: str, check_interval: int = 1200) -> Dict[str, Any]:

    client = get_client()
    batch = client.batches.retrieve(batch_id)

    start_time = time.time()

    while batch.status != "completed":
        if batch.status in ["failed", "cancelled"]:
            logger.error(f"Batch {batch_id} ended with status: {batch.status}")
            raise Exception(f"Batch failed with status: {batch.status}")

        elapsed_time = time.time() - start_time
        logger.info(
            f"Batch {batch_id} status: {batch.status}, elapsed time: {elapsed_time:.2f}s"
        )

        if hasattr(batch, "metrics") and batch.metrics:
            completed = batch.metrics.get("completed", 0)
            total = batch.metrics.get("total", 0)
            if total > 0:
                logger.info(
                    f"Progress: {completed}/{total} ({(completed/total)*100:.2f}%)"
                )

        time.sleep(check_interval)
        batch = client.batches.retrieve(batch_id)

    metrics_file = f"{Config.BATCH_METRICS_DIR}/metrics_it_{batch.metadata['description'].split()[1]}.json"
    metrics_data = {
        "batch_id": batch_id,
        "status": batch.status,
        "created_at": getattr(batch, "created_at", None),
        "started_at": getattr(batch, "started_at", None),
        "completed_at": getattr(batch, "completed_at", None),
        "metrics": getattr(batch, "metrics", {}),
        "error_count": getattr(batch, "error_count", 0),
    }

    with open(metrics_file, "w") as f:
        json.dump(metrics_data, f, indent=2)

    logger.info(f"Batch {batch_id} completed in {time.time() - start_time:.2f}s")
    return batch


def download_batch_results(batch_id: str, iteration: int) -> str:

    client = get_client()

    batch = client.batches.retrieve(batch_id)
    output_file_id = getattr(batch, "output_file_id", None)

    if not output_file_id:
        logger.error(f"No output file ID found for batch {batch_id}")
        output_path = f"{Config.BATCH_OUTPUT_DIR}/it_{iteration}.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(
                json.dumps({"error": f"No output file for batch {batch_id}"}) + "\n"
            )

        logger.info(f"Created empty placeholder file at {output_path}")
        return output_path

    try:
        file_response = client.files.content(output_file_id)
        output_path = f"{Config.BATCH_OUTPUT_DIR}/it_{iteration}.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for line in file_response.text.splitlines():
                f.write(line + "\n")

        logger.info(f"Saved batch results to {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error downloading batch results: {str(e)}")
        output_path = f"{Config.BATCH_OUTPUT_DIR}/it_{iteration}.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"error": str(e)}) + "\n")

        logger.info(f"Created error placeholder file at {output_path}")
        return output_path


def process_output_for_next_iteration(
    output_file: str, iteration: int
) -> List[Dict[str, Any]]:
    """
    processes the output of one iteration to prepare for the next iteration.
    """
    logger.info(
        f"Processing output from iteration {iteration} for iteration {iteration + 1}"
    )

    try:
        with open(output_file, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]

        translation_jobs = []

        if len(lines) == 1 and "error" in lines[0]:
            logger.warning(
                f"Found error placeholder for iteration {iteration}. Using fallback data."
            )
            return _create_fallback_jobs(iteration)

        for item in lines:
            custom_id = item.get("custom_id", "")
            if not custom_id:
                logger.warning(
                    f"Item missing custom_id in iteration {iteration}. Skipping."
                )
                continue

            parts = custom_id.split("_")
            if len(parts) < 4:
                logger.warning(
                    f"Invalid custom_id format: {custom_id} in iteration {iteration}. Skipping."
                )
                continue

            try:
                idx = int(parts[3])
                source_language = parts[0]
                target_language = parts[2]

                if "response" in item.keys():
                    text = item["response"]["body"]["choices"][0]["message"]["content"]
                else:
                    text = item["text"]

                translation_jobs.append(
                    {
                        "id": idx,
                        "text": text,
                        "source_language": target_language,
                        "target_language": source_language,
                    }
                )
            except (IndexError, KeyError, ValueError) as e:
                logger.warning(
                    f"Error processing item in iteration {iteration}: {str(e)}"
                )
                continue

        if not translation_jobs:
            logger.warning(
                f"No valid translation jobs processed for iteration {iteration}. Using fallback data."
            )
            return _create_fallback_jobs(iteration)

        return translation_jobs

    except Exception as e:
        logger.error(f"Error processing output file: {str(e)}")
        return _create_fallback_jobs(iteration)


def _create_fallback_jobs(iteration: int) -> List[Dict[str, Any]]:

    return [
        {
            "id": 1,
            "text": f"This is a fallback text for iteration {iteration} due to processing error.",
            "source_language": "English",
            "target_language": "Spanish",
        }
    ]


def load_jobs_from_iteration_file(iteration_file: str) -> List[Dict[str, Any]]:

    logger.info(f"Loading jobs from iteration file: {iteration_file}")

    try:
        with open(iteration_file, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]

        translation_jobs = []

        for item in lines:
            custom_id = item.get("custom_id", "")
            if not custom_id:
                logger.warning(
                    f"Item missing custom_id in file {iteration_file}. Skipping."
                )
                continue

            parts = custom_id.split("_")
            if len(parts) < 4:
                logger.warning(
                    f"Invalid custom_id format: {custom_id} in file {iteration_file}. Skipping."
                )
                continue

            try:
                idx = int(parts[3])
                source_language = parts[0]
                target_language = parts[2]

                if "response" in item:
                    text = item["response"]["body"]["choices"][0]["message"]["content"]
                else:
                    text = item["text"]

                translation_jobs.append(
                    {
                        "id": idx,
                        "text": text,
                        "source_language": source_language,
                        "target_language": target_language,
                    }
                )
            except (IndexError, KeyError, ValueError) as e:
                logger.warning(
                    f"Error processing item in file {iteration_file}: {str(e)}"
                )
                continue

        if not translation_jobs:
            raise ValueError(f"No valid translation jobs found in {iteration_file}")

        return translation_jobs

    except Exception as e:
        logger.error(f"Error loading jobs from iteration file: {str(e)}")
        raise


def run_translation_pipeline(
    initial_jobs: List[Dict[str, Any]] = None,
    max_iterations: int = 18,
    start_iteration: int = 1,
    start_iteration_file: str = None,
):

    create_directories()

    if start_iteration < 1 or start_iteration > max_iterations:
        raise ValueError(f"start_iteration must be between 1 and {max_iterations}")

    if start_iteration == 1:
        if not initial_jobs:
            raise ValueError("initial_jobs is required when starting from iteration 1")
        current_jobs = initial_jobs
    else:
        if not start_iteration_file:
            raise ValueError(
                "start_iteration_file is required when starting from iteration > 1"
            )
        current_jobs = load_jobs_from_iteration_file(start_iteration_file)

    batch_ids = {}

    for iteration in range(start_iteration, max_iterations + 1):
        logger.info(f"Starting iteration {iteration}/{max_iterations}")

        is_back = Config.is_back_translation(iteration)
        translation_type = "Back Translation" if is_back else "Forward Translation"

        model = Config.get_model_for_iteration(iteration)
        logger.info(f"Iteration {iteration}: {translation_type} using model {model}")

        batch_file = write_jsonl_batch(
            translation_jobs=current_jobs,
            model=model,
            iteration=iteration,
            temperature=0.2,
        )
        batch_id = create_batch(batch_file, iteration)
        batch_ids[iteration] = batch_id

        batch = monitor_batch(batch_id)

        output_file = download_batch_results(batch_id, iteration)

        if iteration < max_iterations:
            current_jobs = process_output_for_next_iteration(output_file, iteration)

        logger.info(f"Completed iteration {iteration}")

    logger.info("Translation pipeline completed successfully")
    return batch_ids


if __name__ == "__main__":
    #  Start from iteration 1 with initial jobs
    # with open("data/commercial_seed.json", "r") as f:
    #     initial_jobs = json.load(f)
    # batch_ids = run_translation_pipeline(initial_jobs, max_iterations=18)

    #  Start from iteration 11 using it_10.jsonl
    try:
        batch_ids = run_translation_pipeline(
            max_iterations=18,
            start_iteration=11,
            start_iteration_file="outputs/batch_output/it_10.jsonl",
        )
        print("Pipeline completed. Batch IDs:", batch_ids)
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise
