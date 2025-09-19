import os
import json
import torch
import orjson
import ijson
from tqdm import tqdm
from typing import List, Dict, Any
import tempfile
import shutil
import logging
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import time

torch.set_float32_matmul_precision("high")

from pipeline.evaluation_metrices.xcomet import evaluate as evaluate_xcomet

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
        output_format: str = "parquet",  # Options: "parquet", "json"
        compression: str = "snappy",  # Parquet compression: "snappy", "gzip", "lz4", "zstd"
    ):
        self.input_file = input_file
        self.output_file = output_file
        self.data = self._load_data()
        self.iteration_num = iteration_num
        self.batch_size = batch_size
        self.include_ref = include_ref
        self.include_bt = include_bt
        self.gpus = gpus
        self.output_format = output_format
        self.compression = compression

        # Convert to DataFrame for efficient parquet operations
        if self.output_format == "parquet":
            self.df = self._dict_to_dataframe(self.data)

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load and return the input data."""
        start_time = time.time()

        # Support both JSON and Parquet input
        file_ext = os.path.splitext(self.input_file)[1].lower()

        if file_ext == ".parquet":
            # Load from parquet
            df = pd.read_parquet(self.input_file)
            data = df.to_dict("records")
        else:
            # Load from JSON using orjson
            with open(self.input_file, "rb") as f:
                data = orjson.loads(f.read())

        load_time = time.time() - start_time
        logger.info(f"Loaded {len(data)} items in {load_time:.2f} seconds")
        return data

    def _dict_to_dataframe(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert list of dictionaries to DataFrame for parquet operations."""
        df = pd.DataFrame(data)

        # Convert object columns to string for better parquet performance
        for col in df.select_dtypes(include=["object"]):
            try:
                # Try to convert to numeric first
                df[col] = pd.to_numeric(df[col])
            except:
                # Keep as string
                df[col] = df[col].astype(str)

        return df

    def _dataframe_to_dict(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert DataFrame back to list of dictionaries for evaluation functions."""
        return df.to_dict("records")

    def _save_data_parquet(self) -> None:
        """Save data to parquet format with high compression and speed."""
        start_time = time.time()
        logger.info("Saving results to parquet...")

        # Convert dict data to DataFrame if needed
        if hasattr(self, "df") and isinstance(self.df, pd.DataFrame):
            df = self.df
        else:
            df = self._dict_to_dataframe(self.data)

        # Save to parquet with optimization
        df.to_parquet(
            self.output_file,
            compression=self.compression,
            engine="pyarrow",
            index=False,
            # Enable row group size optimization for large files
            row_group_size=50000,
            # Use efficient data types
            use_dictionary=True,
        )

        save_time = time.time() - start_time
        file_size_mb = os.path.getsize(self.output_file) / (1024 * 1024)
        logger.info(
            f"Saved {file_size_mb:.1f}MB in {save_time:.2f} seconds ({file_size_mb/save_time:.1f} MB/s)"
        )
        logger.info(f"Output saved at: {self.output_file}")

    def _save_data_json(self) -> None:
        """Save data to JSON format (fallback option)."""
        start_time = time.time()
        logger.info("Saving results to JSON...")

        with open(self.output_file, "wb", buffering=1048576) as f:
            f.write(orjson.dumps(self.data))

        save_time = time.time() - start_time
        file_size_mb = os.path.getsize(self.output_file) / (1024 * 1024)
        logger.info(
            f"Saved {file_size_mb:.1f}MB in {save_time:.2f} seconds ({file_size_mb/save_time:.1f} MB/s)"
        )
        logger.info(f"Output saved at: {self.output_file}")

    def _save_data(self) -> None:
        """Save data in the specified output format."""
        if self.output_format == "parquet":
            self._save_data_parquet()
        else:
            self._save_data_json()

    def _prepare_text_lists(self, data: List[Dict[str, Any]], i: int, j: int) -> tuple:
        """Prepare text lists for evaluation."""
        text_list = []
        reverse_text_list = []

        for data_single in data:
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
        data: List[Dict[str, Any]],
        xcomet_data: List[Dict[str, Any]],
        reverse_xcomet_data: List[Dict[str, Any]],
        i: int,
        j: int,
    ) -> None:
        """Update the data with evaluation results."""
        # For parquet, update the DataFrame directly
        if self.output_format == "parquet" and hasattr(self, "df"):
            # Create mapping dictionaries for faster lookups
            xcomet_scores = {item["id"]: item["xcomet_score"] for item in xcomet_data}
            reverse_xcomet_scores = {
                item["id"]: item["xcomet_score"] for item in reverse_xcomet_data
            }

            # Update DataFrame columns
            self.df[f"xcomet_src_{j}_mt_{i}"] = self.df["id"].map(xcomet_scores)
            if reverse_xcomet_scores:
                self.df[f"xcomet_src_{j}_mt_{i}_bt"] = self.df["id"].map(
                    reverse_xcomet_scores
                )
        else:
            # Original dictionary-based update
            xcomet_lookup = {
                result["id"]: result["xcomet_score"] for result in xcomet_data
            }
            reverse_xcomet_lookup = {
                result["id"]: result["xcomet_score"] for result in reverse_xcomet_data
            }

            for data_single in data:
                if data_single["id"] in xcomet_lookup:
                    data_single[f"xcomet_src_{j}_mt_{i}"] = xcomet_lookup[
                        data_single["id"]
                    ]

                if self.include_bt and data_single["id"] in reverse_xcomet_lookup:
                    data_single[f"xcomet_src_{j}_mt_{i}_bt"] = reverse_xcomet_lookup[
                        data_single["id"]
                    ]

    def run(self) -> None:
        """Run the evaluation pipeline with optimized saving."""
        logger.info(f"Starting evaluation pipeline")
        logger.info(f"Input file: {self.input_file}")
        logger.info(f"Output file: {self.output_file}")

        for i in range(1, self.iteration_num + 1):
            j = 1
            logger.info(f"Starting iteration {i}")

            # Prepare text lists for evaluation
            if self.output_format == "parquet" and hasattr(self, "df"):
                # Convert DataFrame to dict for evaluation
                data_for_eval = self._dataframe_to_dict(self.df)
            else:
                data_for_eval = self.data

            text_list, reverse_text_list = self._prepare_text_lists(data_for_eval, i, j)

            # Run evaluations
            xcomet_data = evaluate_xcomet(
                text_list=text_list, batch_size=self.batch_size, gpus=self.gpus
            )
            reverse_xcomet_data = (
                evaluate_xcomet(
                    text_list=reverse_text_list,
                    batch_size=self.batch_size,
                    gpus=self.gpus,
                )
                if self.include_bt and i != j
                else []
            )

            # Update results
            self._update_results(data_for_eval, xcomet_data, reverse_xcomet_data, i, j)

            # If not using DataFrame, sync the data
            if not (self.output_format == "parquet" and hasattr(self, "df")):
                self.data = data_for_eval

            # Save progress after each iteration
            logger.info(f"Completed iteration {i}, saving results...")
            self._save_data()
            logger.info(f"Results saved for iteration {i}")

        logger.info("All iterations completed!")


def run_eval_pipeline(
    input_file_name="ip_op_mr.json",
    output_file_name="ip_op_mr.parquet",  # Changed default to parquet
    iteration_num: int = 18,
    batch_size: int = 8,
    include_ref=False,
    include_bt=False,
    gpus=1,
    output_format="parquet",  # Changed default to parquet
    compression="snappy",  # Parquet compression
):
    """Main function to run the evaluation pipeline."""
    # Ensure proper file extension
    if output_format == "parquet" and not output_file_name.endswith(".parquet"):
        output_file_name = output_file_name.replace(".json", ".parquet")

    pipeline = EvaluationPipeline(
        input_file=os.path.join("outputs/all_output", input_file_name),
        output_file=os.path.join("outputs/all_eval", output_file_name),
        iteration_num=iteration_num,
        batch_size=batch_size,
        include_ref=include_ref,
        include_bt=include_bt,
        gpus=gpus,
        output_format=output_format,
        compression=compression,
    )
    pipeline.run()


if __name__ == "__main__":
    run_eval_pipeline(
        input_file_name="ip_op_lowdiv_qe.json",
        output_file_name="ip_op_lowdiv_qe.parquet",  # Use parquet for much faster saves
        batch_size=128,
        iteration_num=18,
        include_bt=False,
        gpus=1,
        output_format="parquet",
        compression="snappy",  # Try "zstd" for even better compression
        include_ref=False,
    )
    # low_div_main()
