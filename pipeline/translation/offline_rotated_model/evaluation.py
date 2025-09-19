import os
import json
import torch
import orjson
import ijson  # For streaming JSON processing
from tqdm import tqdm
from typing import List, Dict, Any
import tempfile
import shutil
import mmap
import time
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

torch.set_float32_matmul_precision("high")

# import nltk
from pipeline.evaluation_metrices.xcomet import evaluate as evaluate_xcomet

# from pipeline.evaluation_metrices.similarity import calculate_cosine_similarity

# # Download required NLTK data
# nltk.download("punkt")
# nltk.download("punkt_tab")


class EvaluationPipeline:
    def __init__(
        self,
        input_file: str,
        output_file: str,
        iteration_num: int,
        batch_size: int = 8,
        include_ref: bool = False,
        output_format: str = "parquet",  # Options: "parquet", "json"
        compression: str = "snappy",  # Parquet compression: "snappy", "gzip", "lz4", "zstd"
    ):
        self.input_file = input_file
        self.output_file = output_file
        self.data = self._load_data()
        self.iteration_num = iteration_num
        self.batch_size = batch_size
        self.include_ref = include_ref
        self.output_format = output_format
        self.compression = compression

        # Convert to DataFrame for efficient parquet operations
        if self.output_format == "parquet":
            self.df = self._dict_to_dataframe(self.data)

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load and return the input JSON data."""
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
        print(f"Loaded {len(data)} items in {load_time:.2f} seconds")
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
        print("Saving results to parquet...")

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
        print(
            f"Saved {file_size_mb:.1f}MB in {save_time:.2f} seconds ({file_size_mb/save_time:.1f} MB/s)"
        )
        print(f"Output saved at: {self.output_file}")

    def _save_data_json(self) -> None:
        """Save data to JSON format (fallback option)."""
        start_time = time.time()
        print("Saving results to JSON...")

        with open(self.output_file, "wb", buffering=1048576) as f:
            f.write(orjson.dumps(self.data))

        save_time = time.time() - start_time
        file_size_mb = os.path.getsize(self.output_file) / (1024 * 1024)
        print(
            f"Saved {file_size_mb:.1f}MB in {save_time:.2f} seconds ({file_size_mb/save_time:.1f} MB/s)"
        )
        print(f"Output saved at: {self.output_file}")

    def _save_data(self) -> None:
        """Save data in the specified output format."""
        if self.output_format == "parquet":
            self._save_data_parquet()
        else:
            self._save_data_json()

    def _clear_error_file(self) -> None:
        """Clear the error file if it exceeds 1000 lines."""
        error_file = "errors/evaluation.txt"
        if os.path.exists(error_file):
            with open(error_file, "r") as f:
                lines = f.readlines()
            if len(lines) > 1000:
                with open(error_file, "w") as f:
                    f.write("")

    def _prepare_text_lists(
        self, data_chunk: List[Dict[str, Any]], i: int, j: int
    ) -> tuple:
        """Prepare text lists for evaluation."""
        text_list = []
        reverse_text_list = []

        for data_single in data_chunk:
            self._clear_error_file()

            text_list.append(
                {
                    "id": data_single["id"],
                    "src": data_single[f"input_{j}"],
                    "mt": data_single[f"output_{i}"],
                    **({"ref": data_single["ref"]} if self.include_ref else {}),
                }
            )

            reverse_text_list.append(
                {
                    "id": data_single["id"],
                    "src": data_single["ref"],
                    "mt": data_single[f"input_{i}"],
                    **({"ref": data_single[f"input_{j}"]} if self.include_ref else {}),
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
        # For parquet, update the DataFrame directly
        if self.output_format == "parquet" and hasattr(self, "df"):
            xcomet_scores = {item["id"]: item["xcomet_score"] for item in xcomet_data}
            reverse_xcomet_scores = {
                item["id"]: item["xcomet_score"] for item in reverse_xcomet_data
            }

            self.df[f"xcomet_src_{j}_mt_{i}"] = self.df["id"].map(xcomet_scores)
            if reverse_xcomet_scores:
                self.df[f"xcomet_src_{j}_mt_{i}_bt"] = self.df["id"].map(
                    reverse_xcomet_scores
                )
        else:
            for data_single in data_chunk:
                for result in xcomet_data:
                    if data_single["id"] == result["id"]:
                        data_single[f"xcomet_src_{j}_mt_{i}"] = result["xcomet_score"]

                for result in reverse_xcomet_data:
                    if data_single["id"] == result["id"]:
                        data_single[f"xcomet_src_{j}_mt_{i}_bt"] = result[
                            "xcomet_score"
                        ]

    def run(self) -> None:
        for i in range(1, self.iteration_num + 1):
            j = 1

            if self.output_format == "parquet" and hasattr(self, "df"):
                data_for_eval = self._dataframe_to_dict(self.df)
            else:
                data_for_eval = self.data

            text_list, reverse_text_list = self._prepare_text_lists(data_for_eval, i, j)

            xcomet_data = evaluate_xcomet(
                text_list=text_list, batch_size=self.batch_size
            )
            reverse_xcomet_data = (
                evaluate_xcomet(text_list=reverse_text_list, batch_size=self.batch_size)
                if i != j
                else []
            )

            self._update_results(data_for_eval, xcomet_data, reverse_xcomet_data, i, j)

            if not (self.output_format == "parquet" and hasattr(self, "df")):
                self.data = data_for_eval

            print(f"Completed iteration {i}, saving results...")
            self._save_data()
            print(f"Results saved for iteration {i}")

        print("All iterations completed!")


def run_eval_pipeline(
    input_file_name="ip_op_mr.json",
    output_file_name="ip_op_mr.parquet",
    iteration_num: int = 9,
    batch_size: int = 64,
    include_ref=False,
    output_format="parquet",
    compression="snappy",  # Parquet compression
):
    """Main function to run the evaluation pipeline."""

    if output_format == "parquet" and not output_file_name.endswith(".parquet"):
        output_file_name = output_file_name.replace(".json", ".parquet")

    pipeline = EvaluationPipeline(
        input_file=os.path.join("outputs/all_output", input_file_name),
        output_file=os.path.join("outputs/all_eval", output_file_name),
        iteration_num=iteration_num,
        batch_size=batch_size,
        include_ref=include_ref,
        output_format=output_format,
        compression=compression,
    )
    pipeline.run()


if __name__ == "__main__":
    run_eval_pipeline(
        input_file_name="ip_op_mr_ref.json",
        output_file_name="ip_op_mr_ref.parquet",  # Use parquet for much faster saves
        iteration_num=9,
        batch_size=72,
        include_ref=True,
        output_format="parquet",
        compression="snappy",
    )
