import pandas as pd
from typing import Callable, List, Dict, Any, Tuple
import logging


def evaluate_translations(
    input_csv_path: str,
    evaluation_functions: List[Tuple[Callable, str]],
    output_csv_path: str = None,
    save_intermediate: bool = False,
    batch_size: int = 8,
) -> pd.DataFrame:

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logging.info(f"Reading input data from {input_csv_path}")
    df = pd.read_csv(input_csv_path)

    # Validate required columns
    required_columns = ["src", "mt"]  # Removed 'id' from required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Input CSV missing required columns: {', '.join(missing_columns)}"
        )

    # Filter out rows with invalid text data
    original_len = len(df)
    df = df.dropna(subset=["src", "mt"])  # Remove rows where src or mt is None/NaN
    df = df[
        df["src"].str.strip().astype(bool) & df["mt"].str.strip().astype(bool)
    ]  # Remove empty strings
    filtered_len = len(df)

    if filtered_len < original_len:
        logging.warning(
            f"Filtered out {original_len - filtered_len} rows with invalid text data"
        )
        logging.warning(
            "Invalid data includes rows with None, NaN, or empty strings in 'src' or 'mt' columns"
        )

    if "id" not in df.columns:
        logging.info("ID column not found in input CSV. Assigning sequential IDs.")
        df["id"] = [f"{i+1}" for i in range(len(df))]

    df = df.sort_values(by="id").reset_index(drop=True)

    # Create a copy of the original DataFrame to preserve all columns
    result_df = df.copy()

    # Create a mapping of ID to row index for validation
    id_to_index = {row["id"]: idx for idx, row in df.iterrows()}

    # Create the input data with only the required fields for evaluation
    # This ensures we don't pass unnecessary data to the evaluation functions
    data = [
        {
            "id": row["id"],
            "src": str(row["src"]).strip(),  # Ensure string type and strip whitespace
            "mt": str(row["mt"]).strip(),  # Ensure string type and strip whitespace
            "ref": (
                str(row.get("ref", "")).strip() if "ref" in row else ""
            ),  # Handle ref if present
        }
        for _, row in df.iterrows()
    ]

    # Validate that all text fields are non-empty
    invalid_entries = [
        item
        for item in data
        if not item["src"] or not item["mt"]
        # or (item.get("ref") is not None and not item["ref"])
    ]
    if invalid_entries:
        logging.error(
            f"Found {len(invalid_entries)} entries with empty text after processing"
        )
        raise ValueError(
            "Invalid text data found after processing. Please check the input data."
        )

    # For each evaluation function
    for evaluation_function, name in evaluation_functions:
        logging.info(f"Running evaluation: {name}")

        try:
            # Pass the data and batch size to the evaluation function
            results = evaluation_function(
                data
            )  # batch_size is now handled by the wrapper

            # Handle results based on their format
            if (
                isinstance(results, list)
                and results
                and isinstance(results[0], dict)
                and "id" in results[0]
            ):
                logging.info(f"{name} returned results with IDs")

                result_ids = {item["id"] for item in results if "id" in item}
                missing_ids = result_ids - set(id_to_index.keys())
                if missing_ids:
                    raise ValueError(
                        f"{name} returned results for IDs not in input data: {missing_ids}"
                    )

                scores = [None] * len(data)
                for item in results:
                    if "id" in item:
                        idx = id_to_index[item["id"]]
                        score = None
                        for key, value in item.items():
                            if key.endswith("_score") and key != "id":
                                score = value
                                break

                        if score is None:
                            score = item.get("result", None)

                        scores[idx] = score

                if None in scores:
                    raise ValueError(
                        f"{name} did not return scores for all input items"
                    )

                result_df[name] = scores

            else:
                logging.info(f"{name} returned a list of scores (assuming same order)")
                if len(results) != len(data):
                    raise ValueError(
                        f"Results length ({len(results)}) doesn't match input data length ({len(data)})"
                    )
                result_df[name] = results

            if save_intermediate:
                intermediate_path = f"{input_csv_path.rsplit('.', 1)[0]}-{name}.csv"
                result_df.to_csv(intermediate_path, index=False)
                logging.info(f"Saved intermediate results to {intermediate_path}")

        except Exception as e:
            logging.error(f"Error during {name} evaluation: {str(e)}")
            result_df[name] = pd.NA

    if output_csv_path is None:
        output_csv_path = f"{input_csv_path.rsplit('.', 1)[0]}-all-metrics.csv"

    result_df.to_csv(output_csv_path, index=False)
    logging.info(f"Evaluation complete. Results saved to {output_csv_path}")

    return result_df
