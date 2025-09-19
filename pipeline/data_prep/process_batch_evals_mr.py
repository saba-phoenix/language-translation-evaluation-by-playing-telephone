import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import random


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
    input_file, max_iter=9, project_name="full_noref", df_prefix="xcomet_"
):
    # Read the JSON file
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Get the splits from high diversity data
    train_base_ids, test_base_ids = get_existing_splits()

    # Group by source-target language combination
    lang_groups = defaultdict(list)
    for item in data:
        source_lang = item["source_language"]
        target_lang = item["target_language"]
        lang_pair = f"{source_lang}-{target_lang}"  # Create language pair key
        lang_groups[lang_pair].append(item)

    # Process each language group
    for lang_pair, items in lang_groups.items():
        # Create dataframe
        df_data = []

        for item in items:
            row_data = {"id": int(item["id"])}  # Convert id to integer

            # Add xcomet scores for each iteration
            for i in range(1, max_iter + 1):  # 1 to 9
                score_key = f"xcomet_src_1_mt_{i}"
                if score_key in item:
                    row_data[f"iteration_{i}"] = item[score_key]
                else:
                    print(f"Warning: {score_key} not found for id {item['id']}")
                    row_data[f"iteration_{i}"] = np.nan

            df_data.append(row_data)

        # Create DataFrame
        df = pd.DataFrame(df_data)

        # Debug prints for base IDs
        print(f"\nDebug for {lang_pair} data:")
        print(f"Number of items: {len(df)}")
        print(f"Sample IDs: {df['id'].head().tolist()}")

        # Verify columns exist
        required_columns = [f"iteration_{i}" for i in range(1, max_iter + 1)]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Warning: Missing columns: {missing_columns}")
            # Add missing columns with NaN values
            for col in missing_columns:
                df[col] = np.nan

        # Calculate row means
        df["row_mean"] = df[required_columns].mean(axis=1)

        # Calculate column statistics
        column_stats = {}
        for i in range(1, max_iter + 1):
            col = f"iteration_{i}"
            column_stats[col] = {"mean": df[col].mean(), "std": df[col].std()}

        # Calculate row z-scores
        row_mean_mean = df["row_mean"].mean()
        row_mean_std = df["row_mean"].std()
        df["row_zscore"] = df["row_mean"].apply(
            lambda x: calculate_zscore(x, row_mean_mean, row_mean_std)
        )

        # Calculate expected scores based on row z-scores
        for i in range(1, max_iter + 1):
            col = f"iteration_{i}"
            # Calculate the score in the column's normal distribution that corresponds to the row's z-score
            df[f"{col}_expected"] = df["row_zscore"].apply(
                lambda z: calculate_score_from_zscore(
                    z, column_stats[col]["mean"], column_stats[col]["std"]
                )
            )

        # Create flattened data for CSV
        train_data = []
        test_data = []

        # Get source and target languages from the lang_pair
        source_lang, target_lang = lang_pair.split("-")

        # Process each item
        train_count = 0
        test_count = 0
        for df_idx in df.index:
            row = df.loc[df_idx]
            item_idx = df_idx  # Since we created the DataFrame with continuous indices
            base_id = int(row["id"])  # Keep as integer

            # Use the same split as high diversity data based only on ID
            split = "train" if base_id in train_base_ids else "test"

            if split == "train":
                train_count += 1
            else:
                test_count += 1

            for j in range(1, max_iter + 1):
                translation = {
                    "id": f"id_{int(row['id'])}_it_{j}",  # Convert id to integer
                    "src": items[item_idx]["input_1"],
                    "mt": items[item_idx][f"output_{j}"],
                    "ref": items[item_idx]["ref"],
                    "score": float(row[f"iteration_{j}_expected"]),
                    "source_language": source_lang,
                    "target_language": target_lang,
                    "xcomet_score": float(items[item_idx][f"xcomet_src_1_mt_{j}"]),
                }

                if split == "train":
                    train_data.append(translation)
                else:
                    test_data.append(translation)

        print(f"\nSplit counts for {lang_pair}:")
        print(f"Train items: {train_count}")
        print(f"Test items: {test_count}")

        # Save to CSV
        output_dir = Path("outputs/transformed_df")
        output_dir.mkdir(exist_ok=True)

        # Make sure project directory exists
        project_dir = output_dir / project_name
        project_dir.mkdir(exist_ok=True)

        # Save train data
        train_df = pd.DataFrame(train_data)
        train_csv_path = project_dir / f"{df_prefix}{lang_pair}_train.csv"
        train_df.to_csv(train_csv_path, index=False)
        print(f"Created {train_csv_path}")

        # Save test data
        test_df = pd.DataFrame(test_data)
        test_csv_path = project_dir / f"{df_prefix}{lang_pair}_test.csv"
        test_df.to_csv(test_csv_path, index=False)
        print(f"Created {test_csv_path}")

        # Print summary statistics
        print(f"\nSummary for {lang_pair}:")
        print("Column Statistics:")
        for i in range(1, max_iter + 1):
            col = f"iteration_{i}"
            print(
                f"{col}: mean={column_stats[col]['mean']:.4f}, std={column_stats[col]['std']:.4f}"
            )
        print(f"\nRow Statistics:")
        print(f"Mean of row means: {row_mean_mean:.4f}")
        print(f"Std of row means: {row_mean_std:.4f}")
        print(f"Train/Test split: {len(train_data)}/{len(test_data)}")


if __name__ == "__main__":
    max_iter = 9
    project_name = "model_rotation"
    input_file = "outputs/all_eval/ip_op_mrc_eval_fixed.json"
    df_prefix = "xcomet_"
    process_batch_eval(
        input_file, max_iter=max_iter, project_name=project_name, df_prefix=df_prefix
    )
