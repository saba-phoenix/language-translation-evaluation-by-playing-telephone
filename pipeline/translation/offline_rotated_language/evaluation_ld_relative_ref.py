import os
import json
import torch
import orjson
import ijson
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional
import tempfile
import shutil
import logging
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import time


torch.set_float32_matmul_precision("high")

from pipeline.evaluation_metrices.xcomet import evaluate as evaluate_xcomet
from pipeline.translation.offline_rotated_language.triplets_low_div import (
    main as low_div_main,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EnhancedEvaluationPipeline:
    def __init__(
        self,
        input_file: str,
        output_file: str,
        iteration_num: int,
        batch_size: int = 8,
        include_ref: bool = True,  # Changed default to True as we need references
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

        # Language mapping
        self.l2_positions = [2, 8, 14]
        self.l3_positions = [4, 10, 16]
        self.english_positions = [1, 3, 5, 7, 9, 11, 13, 15, 17]

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

    def _get_language_type(self, position: int) -> str:
        """Return language type based on position."""
        if position in self.l2_positions:
            return "L2"
        elif position in self.l3_positions:
            return "L3"
        elif position in self.english_positions:
            return "English"
        else:
            return "Unknown"

    def _get_mt_ref_pairs(self) -> List[Dict[str, Any]]:
        """Get all MT-reference pairs according to the requirements."""
        pairs = []

        # 1. When input is input_1 and output is L2 or L3
        # For L2 positions: [2, 8, 14]
        pairs.extend(
            [
                {
                    "input_key": "input_1",
                    "mt_key": "output_8",
                    "ref_key": "output_2",
                    "name": "ip_input_1_8_vs_2",
                },
                {
                    "input_key": "input_1",
                    "mt_key": "output_14",
                    "ref_key": "output_2",
                    "name": "ip_input_1_14_vs_2",
                },
                {
                    "input_key": "input_1",
                    "mt_key": "output_14",
                    "ref_key": "output_8",
                    "name": "ip_input_1_14_vs_8",
                },
            ]
        )

        # For L3 positions: [4, 10, 16]
        pairs.extend(
            [
                {
                    "input_key": "input_1",
                    "mt_key": "output_10",
                    "ref_key": "output_4",
                    "name": "ip_input_1_10_vs_4",
                },
                {
                    "input_key": "input_1",
                    "mt_key": "output_16",
                    "ref_key": "output_4",
                    "name": "ip_input_1_16_vs_4",
                },
                {
                    "input_key": "input_1",
                    "mt_key": "output_16",
                    "ref_key": "output_10",
                    "name": "ip_input_1_16_vs_10",
                },
            ]
        )

        # 2. When input is input_1 and output is English
        for pos in [3, 5, 7, 9, 11, 13, 15, 17]:
            pairs.append(
                {
                    "input_key": "input_1",
                    "mt_key": f"output_{pos}",
                    "ref_key": "output_1",
                    "name": f"ip_input_1_{pos}_vs_1",
                }
            )

        # 3. When input is output_1 (English) and output is L2 or L3
        # For L2 positions
        pairs.extend(
            [
                {
                    "input_key": "output_1",
                    "mt_key": "output_8",
                    "ref_key": "output_2",
                    "name": "ip_output_1_8_vs_2",
                },
                {
                    "input_key": "output_1",
                    "mt_key": "output_14",
                    "ref_key": "output_2",
                    "name": "ip_output_1_14_vs_2",
                },
                {
                    "input_key": "output_1",
                    "mt_key": "output_14",
                    "ref_key": "output_8",
                    "name": "ip_output_1_14_vs_8",
                },
            ]
        )

        # For L3 positions
        pairs.extend(
            [
                {
                    "input_key": "output_1",
                    "mt_key": "output_10",
                    "ref_key": "output_4",
                    "name": "ip_output_1_10_vs_4",
                },
                {
                    "input_key": "output_1",
                    "mt_key": "output_16",
                    "ref_key": "output_4",
                    "name": "ip_output_1_16_vs_4",
                },
                {
                    "input_key": "output_1",
                    "mt_key": "output_16",
                    "ref_key": "output_10",
                    "name": "ip_output_1_16_vs_10",
                },
            ]
        )

        return pairs

    def _prepare_text_lists(
        self, data: List[Dict[str, Any]], pair: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Prepare text lists for evaluation based on specific MT-reference pair."""
        text_list = []

        input_key = pair["input_key"]
        mt_key = pair["mt_key"]
        ref_key = pair["ref_key"]

        for data_single in data:
            if (
                input_key in data_single
                and mt_key in data_single
                and ref_key in data_single
            ):
                text_list.append(
                    {
                        "id": data_single["id"],
                        "src": data_single[input_key],
                        "mt": data_single[mt_key],
                        "ref": data_single[ref_key],
                    }
                )

        return text_list

    def _update_results(
        self,
        pair: Dict[str, Any],
        xcomet_data: List[Dict[str, Any]],
    ) -> None:
        """Update the data with evaluation results."""
        # Create column name based on the pair name
        column_name = f"xcomet_{pair['name']}"

        # For parquet, update the DataFrame directly
        if self.output_format == "parquet" and hasattr(self, "df"):
            # Create mapping dictionaries for faster lookups
            xcomet_scores = {item["id"]: item["xcomet_score"] for item in xcomet_data}

            # Update DataFrame columns
            self.df[column_name] = self.df["id"].map(xcomet_scores)
        else:
            # Original dictionary-based update
            xcomet_lookup = {
                result["id"]: result["xcomet_score"] for result in xcomet_data
            }

            for data_single in self.data:
                if data_single["id"] in xcomet_lookup:
                    data_single[column_name] = xcomet_lookup[data_single["id"]]

    def run(self) -> None:
        """Run the enhanced evaluation pipeline with specified MT-reference pairs."""
        logger.info(f"Starting enhanced evaluation pipeline")
        logger.info(f"Input file: {self.input_file}")
        logger.info(f"Output file: {self.output_file}")

        # Get all MT-reference pairs
        pairs = self._get_mt_ref_pairs()
        logger.info(f"Total evaluation pairs: {len(pairs)}")

        # Prepare data for evaluation
        if self.output_format == "parquet" and hasattr(self, "df"):
            data_for_eval = self._dataframe_to_dict(self.df)
        else:
            data_for_eval = self.data

        # Run evaluations for each pair
        for idx, pair in enumerate(pairs):
            logger.info(f"Processing pair {idx+1}/{len(pairs)}: {pair['name']}")

            # Prepare text lists for evaluation
            text_list = self._prepare_text_lists(data_for_eval, pair)

            if not text_list:
                logger.warning(
                    f"No valid data found for pair: {pair['name']}, skipping..."
                )
                continue

            logger.info(f"Evaluating {len(text_list)} samples for pair: {pair['name']}")

            # Run XCOMET evaluation
            xcomet_data = evaluate_xcomet(
                text_list=text_list, batch_size=self.batch_size, gpus=self.gpus
            )

            # Update results
            self._update_results(pair, xcomet_data)

            # Save progress periodically (every 5 pairs or at the end)
            if (idx + 1) % 5 == 0 or idx == len(pairs) - 1:
                logger.info(f"Saving intermediate results after pair {idx+1}...")
                self._save_data()

        logger.info("All evaluations completed!")


def run_enhanced_eval_pipeline(
    input_file_name="ip_op_mr.json",
    output_file_name="ip_op_mr_enhanced_eval.parquet",
    iteration_num: int = 18,
    batch_size: int = 8,
    include_ref=True,
    include_bt=False,
    gpus=1,
    output_format="parquet",
    compression="snappy",
):
    """Main function to run the enhanced evaluation pipeline."""
    # Ensure proper file extension
    if output_format == "parquet" and not output_file_name.endswith(".parquet"):
        output_file_name = output_file_name.replace(".json", ".parquet")

    pipeline = EnhancedEvaluationPipeline(
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
    run_enhanced_eval_pipeline(
        input_file_name="ip_op_lowdiv_qe.json",
        output_file_name="ip_op_lowdiv_relative_ref.parquet",
        batch_size=64,  # Adjusted batch size for better efficiency
        iteration_num=18,
        include_bt=False,
        gpus=1,
        output_format="parquet",
        compression="snappy",
        include_ref=True,
    )
    # Optional: Run low diversity evaluation as well
    low_div_main()
