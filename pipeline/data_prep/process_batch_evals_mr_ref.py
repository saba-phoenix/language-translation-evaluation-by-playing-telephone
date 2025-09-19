import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import random

from data_prep.combine_training_data import combine_training_data


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
    df = pd.read_parquet(input_file)
    data = df.to_dict("records")

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

        # Create dataframe for back translations (only for iterations 2-9)
        df_bt_data = []

        for item in items:
            row_data = {"id": int(item["id"])}  # Convert id to integer

            # Add xcomet scores for each iteration (back-translation)
            # Note: no back-translation for i=1
            for i in range(2, max_iter + 1):  # 2 to 9
                score_key = f"xcomet_src_1_mt_{i}_bt"
                if score_key in item:
                    row_data[f"iteration_{i}"] = item[score_key]
                else:
                    print(f"Warning: {score_key} not found for id {item['id']}")
                    row_data[f"iteration_{i}"] = np.nan

            df_bt_data.append(row_data)

        # Create DataFrame for back translations
        df_bt = pd.DataFrame(df_bt_data)

        # Debug prints for base IDs
        print(f"\nDebug for {lang_pair} data:")
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

        # For back-translations, we only have iterations 2-9
        required_columns_bt = [f"iteration_{i}" for i in range(2, max_iter + 1)]
        missing_columns_bt = [
            col for col in required_columns_bt if col not in df_bt.columns
        ]
        if missing_columns_bt:
            print(
                f"Warning: Missing columns in back-translation df: {missing_columns_bt}"
            )
            # Add missing columns with NaN values
            for col in missing_columns_bt:
                df_bt[col] = np.nan

        # Calculate row means for both
        df["row_mean"] = df[required_columns].mean(axis=1)
        df_bt["row_mean"] = df_bt[required_columns_bt].mean(axis=1)

        # Calculate column statistics for both
        column_stats = {}
        column_stats_bt = {}

        for i in range(1, max_iter + 1):
            col = f"iteration_{i}"
            column_stats[col] = {"mean": df[col].mean(), "std": df[col].std()}

            # Back-translation statistics only for iterations 2-9
            if i >= 2:
                column_stats_bt[col] = {
                    "mean": df_bt[col].mean(),
                    "std": df_bt[col].std(),
                }

        # Calculate row z-scores for both
        row_mean_mean = df["row_mean"].mean()
        row_mean_std = df["row_mean"].std()
        df["row_zscore"] = df["row_mean"].apply(
            lambda x: calculate_zscore(x, row_mean_mean, row_mean_std)
        )

        row_mean_mean_bt = df_bt["row_mean"].mean()
        row_mean_std_bt = df_bt["row_mean"].std()
        df_bt["row_zscore"] = df_bt["row_mean"].apply(
            lambda x: calculate_zscore(x, row_mean_mean_bt, row_mean_std_bt)
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
            # Back translations (only for iterations 2-9)
            if i >= 2:
                df_bt[f"{col}_expected"] = df_bt["row_zscore"].apply(
                    lambda z: calculate_score_from_zscore(
                        z, column_stats_bt[col]["mean"], column_stats_bt[col]["std"]
                    )
                )

        # Create flattened data for CSV
        train_data = []
        test_data = []

        # Get source and target languages from the lang_pair
        source_lang, target_lang = lang_pair.split("-")

        # Process each item for forward translations
        train_count = 0
        test_count = 0

        for df_idx in range(len(items)):  # Use range to ensure proper indexing
            item = items[df_idx]  # Get the actual item from the list
            base_id = int(item["id"])  # Get ID from the original item

            # Use the same split as high diversity data based only on ID
            split = "train" if base_id in train_base_ids else "test"

            if split == "train":
                train_count += 1
            else:
                test_count += 1

            for j in range(1, max_iter + 1):
                # Forward translation
                forward_translation = {
                    "id": f"id_{base_id}_it_{j}",
                    "src": item["input_1"],
                    "mt": item[f"output_{j}"],
                    "ref": item["ref"],
                    "score": float(df.loc[df_idx, f"iteration_{j}_expected"]),
                    "source_language": source_lang,
                    "target_language": target_lang,
                    "xcomet_score": float(item[f"xcomet_src_1_mt_{j}"]),
                }

                # Back translation (with flipped languages) - only for iterations 2-9
                if j >= 2:
                    back_translation = {
                        "id": f"id_{base_id}_bt_{j}",
                        "src": item["ref"],  # ref becomes source
                        "mt": item[f"input_{j}"],  # input_j becomes mt
                        "ref": item["input_1"],  # input_1 becomes reference
                        "score": float(
                            df_bt.loc[df_idx, f"iteration_{j}_expected"]
                        ),  # z-scored back-translation score
                        "source_language": target_lang,  # Flipped: target becomes source
                        "target_language": source_lang,  # Flipped: source becomes target
                        "xcomet_score": float(
                            item[f"xcomet_src_1_mt_{j}_bt"]
                        ),  # back-translation xcomet score
                    }

                    if split == "train":
                        train_data.append(forward_translation)
                        train_data.append(back_translation)
                    else:
                        test_data.append(forward_translation)
                        # test_data.append(back_translation)
                else:
                    # For iteration 1, only add forward translation (no back-translation)
                    if split == "train":
                        train_data.append(forward_translation)
                    else:
                        test_data.append(forward_translation)

        print(f"\nSplit counts for {lang_pair}:")
        print(
            f"Train items: {len([x for x in train_data if x['source_language'] == source_lang])} forward, {len([x for x in train_data if x['source_language'] == target_lang])} back"
        )
        print(
            f"Test items: {len([x for x in test_data if x['source_language'] == source_lang])} forward, {len([x for x in test_data if x['source_language'] == target_lang])} back"
        )

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
        print("Column Statistics (Forward):")
        for i in range(1, max_iter + 1):
            col = f"iteration_{i}"
            print(
                f"{col}: mean={column_stats[col]['mean']:.4f}, std={column_stats[col]['std']:.4f}"
            )
        print(f"\nColumn Statistics (Back):")
        for i in range(2, max_iter + 1):
            col = f"iteration_{i}"
            print(
                f"{col}: mean={column_stats_bt[col]['mean']:.4f}, std={column_stats_bt[col]['std']:.4f}"
            )
        print(f"\nRow Statistics (Forward):")
        print(f"Mean of row means: {row_mean_mean:.4f}")
        print(f"Std of row means: {row_mean_std:.4f}")
        print(f"\nRow Statistics (Back):")
        print(f"Mean of row means: {row_mean_mean_bt:.4f}")
        print(f"Std of row means: {row_mean_std_bt:.4f}")
        print(f"Train/Test split: {len(train_data)}/{len(test_data)}")


if __name__ == "__main__":
    max_iter = 9
    project_name = "model_rotation_ref"
    input_file = "outputs/all_eval/ip_op_mr_ref.parquet"
    df_prefix = "xcomet_"
    process_batch_eval(
        input_file, max_iter=max_iter, project_name=project_name, df_prefix=df_prefix
    )
    combine_training_data(project_name=project_name)
