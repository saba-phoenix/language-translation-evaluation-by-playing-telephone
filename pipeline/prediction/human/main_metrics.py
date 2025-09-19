import pandas as pd
from typing import List, Dict, Any
from pipeline.evaluation_metrices.bleu import evaluate as evaluate_bleu
from pipeline.evaluation_metrices.chrf import evaluate as evaluate_chrf
from pipeline.evaluation_metrices.ter import evaluate as evaluate_ter
from pipeline.evaluation_metrices.meteor import evaluate as evaluate_meteor
from pipeline.prediction.human.eval import evaluate_translations
import logging

# Global configuration
BATCH_SIZE = 128  # Unified batch size for all models

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def filter_data_for_traditional_metrics(input_csv_path: str) -> pd.DataFrame:
    """
    Filter the input data to remove rows with NaN or empty values, especially for ref column.

    Args:
        input_csv_path: Path to the input CSV file

    Returns:
        Filtered DataFrame with valid data for traditional metrics
    """
    # Read the input CSV
    logging.info(f"Reading input data from {input_csv_path}")
    df = pd.read_csv(input_csv_path)

    original_len = len(df)
    logging.info(f"Original dataset size: {original_len}")

    # Filter out rows with NaN values in required columns
    df = df.dropna(subset=["src", "mt", "ref"])
    after_nan_filter = len(df)
    logging.info(f"After NaN filter: {after_nan_filter} rows")

    # Filter out rows with empty strings or whitespace-only strings
    df = df[
        df["src"].str.strip().astype(bool)
        & df["mt"].str.strip().astype(bool)
        & df["ref"].str.strip().astype(bool)
    ]
    after_empty_filter = len(df)
    logging.info(f"After empty string filter: {after_empty_filter} rows")

    # Reset index after filtering
    df = df.reset_index(drop=True)

    # Add ID column if it doesn't exist
    if "id" not in df.columns:
        logging.info("ID column not found. Assigning sequential IDs.")
        df["id"] = [f"{i+1}" for i in range(len(df))]

    # Sort DataFrame by ID to ensure consistent ordering
    df = df.sort_values(by="id").reset_index(drop=True)

    filtered_count = original_len - len(df)
    if filtered_count > 0:
        logging.warning(f"Filtered out {filtered_count} rows with invalid data")
        logging.warning(
            "Invalid data includes rows with NaN, empty strings, or whitespace-only strings in 'src', 'mt', or 'ref' columns"
        )

    logging.info(f"Final dataset size for traditional metrics: {len(df)} rows")
    return df


def create_eval_wrapper(eval_func, use_ref: bool = True, batch_size: int = BATCH_SIZE):
    """
    Creates a wrapper function that formats the input data appropriately for the evaluation function.

    Args:
        eval_func: The original evaluation function
        use_ref: Whether to include reference text in the input data (True for traditional metrics)
        batch_size: Batch size for model evaluation (defaults to global BATCH_SIZE)

    Returns:
        A wrapped function that handles the data formatting
    """

    def wrapper(data: List[Dict[str, Any]]):
        if use_ref:
            # For traditional metrics that need reference text
            formatted_data = [
                {
                    "id": item["id"],
                    "src": item["src"],
                    "mt": item["mt"],
                    "ref": item.get("ref", ""),
                }
                for item in data
            ]
        else:
            # For models that only need source and machine translation
            formatted_data = [
                {"id": item["id"], "src": item["src"], "mt": item["mt"]}
                for item in data
            ]
        return eval_func(formatted_data)

    return wrapper


# Define evaluation functions for traditional metrics
evaluation_functions = [
    (create_eval_wrapper(evaluate_bleu, use_ref=True), "bleu"),
    (create_eval_wrapper(evaluate_chrf, use_ref=True), "chrf"),
    (create_eval_wrapper(evaluate_ter, use_ref=True), "ter"),
    (create_eval_wrapper(evaluate_meteor, use_ref=True), "meteor"),
]

# Filter data for traditional metrics
filtered_df = filter_data_for_traditional_metrics("benchmark_predictions/mqm_6.csv")

# Save filtered data to a temporary file for evaluation
temp_input_path = "benchmark_predictions/mqm_6_filtered.csv"
filtered_df.to_csv(temp_input_path, index=False)
logging.info(f"Saved filtered data to {temp_input_path}")

# Run the evaluation on filtered data
results_df = evaluate_translations(
    input_csv_path=temp_input_path,
    evaluation_functions=evaluation_functions,
    save_intermediate=False,
    output_csv_path="benchmark_predictions/mqm_traditional_metrics.csv",
    batch_size=BATCH_SIZE,  # Pass the unified batch size
)

# Display results
print("Traditional Metrics Evaluation Results:")
print(results_df.head())

# Display summary statistics
print("\nSummary Statistics:")
for metric in ["bleu", "chrf", "ter", "meteor"]:
    if metric in results_df.columns:
        print(f"{metric.upper()}:")
        print(f"  Mean: {results_df[metric].mean():.4f}")
        print(f"  Std:  {results_df[metric].std():.4f}")
        print(f"  Min:  {results_df[metric].min():.4f}")
        print(f"  Max:  {results_df[metric].max():.4f}")
        print()

# Clean up temporary file
import os

if os.path.exists(temp_input_path):
    os.remove(temp_input_path)
    logging.info(f"Removed temporary file: {temp_input_path}")
