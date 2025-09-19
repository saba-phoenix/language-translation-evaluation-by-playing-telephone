import os
import orjson
import ijson
from tqdm import tqdm
from typing import List, Dict, Any, Generator
import tempfile
import shutil
import logging

from pipeline.evaluation_metrices.xcomet import evaluate as evaluate_xcomet
from pipeline.translation.offline_rotated_language.triplets_low_div import (
    main as low_div_main,
)

# from pipeline.evaluation_metrices.similarity import calculate_cosine_similarity

# # Download required NLTK data
# nltk.download("punkt")
# nltk.download("punkt_tab")

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EvaluationPipeline:
    def __init__(
        self,
        input_file: str,
        output_file: str,
        iteration_num: int,
        batch_size: int = 8,
        include_ref: bool = False,
        include_bt: bool = False,
        gpus: int = 1,
        chunk_size: int = 1000,  # Number of items to process at once
        limit: int = None,  # Limit number of items to process
    ):
        self.input_file = input_file
        self.output_file = output_file
        self.iteration_num = iteration_num
        self.batch_size = batch_size
        self.include_ref = include_ref
        self.include_bt = include_bt
        self.gpus = gpus
        self.chunk_size = chunk_size
        self.limit = limit
        self.temp_dir = tempfile.mkdtemp()
        self.temp_file = os.path.join(self.temp_dir, "temp_output.json")

    def __del__(self):
        """Cleanup temporary files"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _stream_data(self) -> Generator[Dict[str, Any], None, None]:
        """Stream data from input file using ijson"""
        count = 0
        with open(self.input_file, "rb") as f:
            parser = ijson.items(f, "item")
            for item in parser:
                if self.limit is not None and count >= self.limit:
                    break
                count += 1
                yield item

    def _process_chunk(
        self, chunk: List[Dict[str, Any]], i: int, j: int
    ) -> List[Dict[str, Any]]:
        """Process a chunk of data"""
        # Prepare text lists for evaluation
        text_list, reverse_text_list = self._prepare_text_lists(chunk, i, j)

        # Run evaluations
        xcomet_data = evaluate_xcomet(
            text_list=text_list, batch_size=self.batch_size, gpus=self.gpus
        )

        if self.include_bt and i != j:
            reverse_xcomet_data = evaluate_xcomet(
                text_list=reverse_text_list, batch_size=self.batch_size, gpus=self.gpus
            )
        else:
            reverse_xcomet_data = []

        # Update results
        self._update_results(chunk, xcomet_data, reverse_xcomet_data, i, j)
        return chunk

    def _prepare_text_lists(
        self, data_chunk: List[Dict[str, Any]], i: int, j: int
    ) -> tuple:
        """Prepare text lists for evaluation."""
        text_list = []
        reverse_text_list = []

        for data_single in data_chunk:
            text_list.append(
                {
                    "id": data_single["id"],
                    "src": data_single[f"input_{j}"],
                    "mt": data_single[f"output_{i}"],
                    **({"ref": data_single["ref"]} if self.include_ref else {}),
                }
            )

            if self.include_bt:
                reverse_text_list.append(
                    {
                        "id": data_single["id"],
                        "src": data_single[f"input_{j}"],
                        "mt": data_single[f"input_{i}"],
                    }
                )

        return text_list, reverse_text_list

    def _update_results(
        self,
        data_chunk: List[Dict[str, Any]],
        xcomet_data: List[Dict[str, Any]],
        reverse_xcomet_data: List[Dict[str, Any]],
        i: int,
        j: int,
    ) -> None:
        """Update the data with evaluation results."""
        # Create a lookup dictionary for faster access
        xcomet_lookup = {result["id"]: result["xcomet_score"] for result in xcomet_data}
        reverse_xcomet_lookup = {
            result["id"]: result["xcomet_score"] for result in reverse_xcomet_data
        }

        for data_single in data_chunk:
            # Process forward evaluation results
            if data_single["id"] in xcomet_lookup:
                data_single[f"xcomet_src_{j}_mt_{i}"] = xcomet_lookup[data_single["id"]]

            # Process backward evaluation results if enabled
            if self.include_bt and data_single["id"] in reverse_xcomet_lookup:
                data_single[f"xcomet_src_{j}_mt_{i}_bt"] = reverse_xcomet_lookup[
                    data_single["id"]
                ]

    def _save_chunk(self, chunk: List[Dict[str, Any]], is_first_chunk: bool = False):
        """Save a chunk of processed data"""
        mode = "wb" if is_first_chunk else "ab"
        try:
            with open(self.temp_file, mode) as f:
                if not is_first_chunk:
                    f.write(b",")
                f.write(orjson.dumps(chunk))
            logger.info(f"Successfully saved chunk of size {len(chunk)} to temp file")
        except Exception as e:
            logger.error(f"Error saving chunk: {str(e)}")
            raise

    def run(self) -> None:
        """Run the evaluation pipeline with streaming and chunking."""
        logger.info(f"Starting evaluation pipeline")
        logger.info(f"Input file: {self.input_file}")
        logger.info(f"Output file: {self.output_file}")
        logger.info(f"Temp file: {self.temp_file}")

        # Initialize output file
        try:
            with open(self.temp_file, "wb") as f:
                f.write(b"[")
            logger.info("Initialized output file")
        except Exception as e:
            logger.error(f"Error initializing output file: {str(e)}")
            raise

        # Load all data first to preserve state between iterations
        all_data = list(self._stream_data())
        logger.info(f"Loaded {len(all_data)} items from input file")

        for i in range(1, self.iteration_num + 1):
            j = 1
            chunk = []
            is_first_chunk = True
            logger.info(f"Starting iteration {i}")

            # Process data in chunks
            for item in tqdm(all_data, desc=f"Processing iteration {i}"):
                chunk.append(item)

                if len(chunk) >= self.chunk_size:
                    logger.info(f"Processing chunk of size {len(chunk)}")
                    processed_chunk = self._process_chunk(chunk, i, j)
                    self._save_chunk(processed_chunk, is_first_chunk)
                    is_first_chunk = False
                    chunk = []

            # Process remaining items
            if chunk:
                logger.info(f"Processing final chunk of size {len(chunk)}")
                processed_chunk = self._process_chunk(chunk, i, j)
                self._save_chunk(processed_chunk, is_first_chunk)

            # Close the JSON array for this iteration
            with open(self.temp_file, "ab") as f:
                f.write(b"]")

            # Save the current state to the output file
            try:
                shutil.copy2(self.temp_file, self.output_file)
                logger.info(f"Saved progress after iteration {i} to {self.output_file}")
            except Exception as e:
                logger.error(f"Error saving progress after iteration {i}: {str(e)}")
                raise

            # Reopen the temp file for the next iteration
            with open(self.temp_file, "wb") as f:
                f.write(b"[")
            logger.info(f"Completed iteration {i}")

        # Clean up the temp file
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)
            logger.info("Cleaned up temporary file")


def run_eval_pipeline(
    input_file_name="ip_op_mr.json",
    output_file_name="ip_op_mr.json",
    iteration_num: int = 18,
    batch_size: int = 8,
    include_ref=False,
    include_bt=False,
    gpus=1,
    chunk_size=1000,
    limit: int = None,  # Limit number of items to process
):
    """Main function to run the evaluation pipeline."""
    pipeline = EvaluationPipeline(
        input_file=os.path.join("outputs/all_output", input_file_name),
        output_file=os.path.join("outputs/all_eval", output_file_name),
        iteration_num=iteration_num,
        batch_size=batch_size,
        include_ref=include_ref,
        include_bt=include_bt,
        gpus=gpus,
        chunk_size=chunk_size,
        limit=limit,
    )
    pipeline.run()


if __name__ == "__main__":
    run_eval_pipeline(
        input_file_name="ip_op_lowdiv_dir.json",
        output_file_name="ip_op_lowdiv_dir_sample.json",
        batch_size=8,
        iteration_num=18,
        include_bt=False,
        gpus=1,
        chunk_size=58000,  # Reduced chunk size
        limit=8,  # Process only 100 items for testing
    )
    low_div_main()
