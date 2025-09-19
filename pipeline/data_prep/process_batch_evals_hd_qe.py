import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import random
from pipeline.utils.data_utils.rotated_lang import high_div_triplets
from data_prep.combine_training_data import combine_training_data
from pipeline.translation.offline_rotated_language.utils.set_seq_from_lang_pair import (
    set_seq_from_lang_pair,
)


def calculate_zscore(value, mean, std):
    return (value - mean) / std if std != 0 else 0


def calculate_score_from_zscore(zscore, mean, std):
    return mean + (zscore * std)


def get_existing_splits():
    """Read existing high diversity splits from the ID files."""
    # Read train and test IDs from the text files
    with open("train_ids_hd.txt", "r") as f:
        train_base_ids = set(map(int, f.read().splitlines()))
    with open("test_ids_hd.txt", "r") as f:
        test_base_ids = set(map(int, f.read().splitlines()))

    print(f"\nDebug for splits:")
    print(f"Number of train IDs: {len(train_base_ids)}")
    print(f"Number of test IDs: {len(test_base_ids)}")
    print(f"Sample train IDs: {sorted(list(train_base_ids))[:5]}")
    print(f"Sample test IDs: {sorted(list(test_base_ids))[:5]}")

    return train_base_ids, test_base_ids


def process_batch_eval(
    input_file,
    max_iter=18,
    project_name="language_rotation_ld_qe",
    df_prefix="xcomet_",
    triplets=high_div_triplets,
):
    df = pd.read_parquet(input_file)
    data = df.to_dict("records")

    # Get the splits from high diversity data
    train_base_ids, test_base_ids = get_existing_splits()

    # Group by source-target language combination
    # Group by target language
    lang_groups = defaultdict(list)
    for item in data:
        if item["source_language"] != "English":
            source_lang = item["source_language"]
        else:
            source_lang = item["target_language"]
        lang_groups[source_lang].append(item)

    # Process each language group
    for lang, items in lang_groups.items():
        # Create dataframe for forward translations
        df_data = []

        for item in items:
            row_data = {"id": int(item["id"])}  # Convert id to integer

            # Add xcomet scores for each iteration (forward)
            for i in range(1, max_iter + 1):  # 1 to 9
                score_key = f"xcomet_src_1_mt_{i}"
                if score_key in item:
                    row_data[f"iteration_{i}"] = item[score_key]
                else:
                    print(f"Warning: {score_key} not found for id {item['id']}")
                    row_data[f"iteration_{i}"] = np.nan

            df_data.append(row_data)

        # Create DataFrame for forward translations
        df = pd.DataFrame(df_data)

        # Debug prints for base IDs
        print(f"\nDebug for {lang} data:")
        print(f"Number of items: {len(df)}")
        print(f"Sample IDs: {df['id'].head().tolist()}")

        # Verify columns exist for both dataframes
        required_columns = [f"iteration_{i}" for i in range(1, max_iter + 1)]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Warning: Missing columns in forward df: {missing_columns}")
            # Add missing columns with NaN values
            for col in missing_columns:
                df[col] = np.nan

        # Calculate row means for both
        df["row_mean"] = df[required_columns].mean(axis=1)

        # Calculate column statistics for both
        column_stats = {}

        for i in range(1, max_iter + 1):
            col = f"iteration_{i}"
            column_stats[col] = {"mean": df[col].mean(), "std": df[col].std()}

        # Calculate row z-scores for both
        row_mean_mean = df["row_mean"].mean()
        row_mean_std = df["row_mean"].std()
        df["row_zscore"] = df["row_mean"].apply(
            lambda x: calculate_zscore(x, row_mean_mean, row_mean_std)
        )

        # Calculate expected scores based on row z-scores for both
        for i in range(1, max_iter + 1):
            col = f"iteration_{i}"
            # Forward translations
            df[f"{col}_expected"] = df["row_zscore"].apply(
                lambda z: calculate_score_from_zscore(
                    z, column_stats[col]["mean"], column_stats[col]["std"]
                )
            )

        # Create flattened data for CSV
        train_data = []
        test_data = []

        # Get source and target languages from the lang_pair
        source_lang = lang

        # Process each item for forward translations
        train_count = 0
        test_count = 0

        for df_idx in range(len(items)):  # Use range to ensure proper indexing
            item = items[df_idx]  # Get the actual item from the list
            base_id = int(item["id"])  # Get ID from the original item
            row = df.loc[df_idx]
            item_idx = df_idx  # Since we created the DataFrame with continuous indices

            # Use the same split as high diversity data based only on ID
            split = "train" if base_id in train_base_ids else "test"

            if split == "train":
                train_count += 1
            else:
                test_count += 1

            for j in range(1, max_iter + 1):
                sequence = set_seq_from_lang_pair(
                    "_",
                    source_lang,
                    triplets,
                    is_direct=False,
                )
                target_language = sequence[((j - 1) + len(sequence)) % len(sequence)]
                forward_translation = {
                    "id": f"id_{int(row['id'])}_it_{j}",  # Convert id to integer
                    "src": items[item_idx]["input_1"],
                    "mt": items[item_idx][f"output_{j}"],
                    # "ref": items[item_idx]["ref"],
                    "score": max(0.0, min(float(row[f"iteration_{j}_expected"]), 1.0)),
                    "source_language": source_lang,
                    "target_language": target_language,
                    "xcomet_score": float(items[item_idx][f"xcomet_src_1_mt_{j}"]),
                }

                if split == "train":
                    train_data.append(forward_translation)
                else:
                    test_data.append(forward_translation)

        print(f"\nSplit counts for {lang}:")
        print(
            f"Train items: {len([x for x in train_data if x['source_language'] == source_lang])} forward"
        )
        print(
            f"Test items: {len([x for x in test_data if x['source_language'] == source_lang])} forward"
        )

        # Save to CSV
        output_dir = Path("outputs/transformed_df")
        output_dir.mkdir(exist_ok=True)

        # Make sure project directory exists
        project_dir = output_dir / project_name
        project_dir.mkdir(exist_ok=True)

        # Save train data
        train_df = pd.DataFrame(train_data)
        train_csv_path = project_dir / f"{df_prefix}{lang}_train.csv"
        train_df.to_csv(train_csv_path, index=False)
        print(f"Created {train_csv_path}")

        # Save test data
        test_df = pd.DataFrame(test_data)
        test_csv_path = project_dir / f"{df_prefix}{lang}_test.csv"
        test_df.to_csv(test_csv_path, index=False)
        print(f"Created {test_csv_path}")

        # Print summary statistics
        print(f"\nSummary for {lang}:")
        print("Column Statistics (Forward):")
        for i in range(1, max_iter + 1):
            col = f"iteration_{i}"
            print(
                f"{col}: mean={column_stats[col]['mean']:.4f}, std={column_stats[col]['std']:.4f}"
            )
        print(f"\nRow Statistics (Forward):")
        print(f"Mean of row means: {row_mean_mean:.4f}")
        print(f"Std of row means: {row_mean_std:.4f}")
        print(f"Train/Test split: {len(train_data)}/{len(test_data)}")


if __name__ == "__main__":
    max_iter = 18
    project_name = "language_rotation_hd_qe"
    input_file = "outputs/all_eval/ip_op_highdiv_qe.parquet"
    df_prefix = "xcomet_"
    triplets = high_div_triplets
    process_batch_eval(
        input_file,
        max_iter=max_iter,
        project_name=project_name,
        df_prefix=df_prefix,
        triplets=triplets,
    )
    combine_training_data(project_name=project_name)
