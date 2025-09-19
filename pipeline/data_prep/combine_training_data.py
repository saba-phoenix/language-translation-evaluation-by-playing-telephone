import pandas as pd
from pathlib import Path


def combine_training_data(project_name="model_rotation"):
    # Path to the directory containing the training files
    data_dir = Path("outputs/transformed_df") / project_name

    # List to store all training DataFrames
    train_dfs = []

    # Read all training CSV files
    for train_file in data_dir.glob("xcomet_*_train.csv"):
        print(f"Reading {train_file}")
        df = pd.read_csv(train_file)
        train_dfs.append(df)

    # Combine all DataFrames
    combined_df = pd.concat(train_dfs, ignore_index=True)

    # Save the combined DataFrame
    output_path = data_dir / "combined_training_data.csv"
    combined_df.to_csv(output_path, index=False)
    print(f"\nCombined training data saved to: {output_path}")
    print(f"Total number of samples: {len(combined_df)}")
    print("\nLanguage pairs in the combined dataset:")
    print(combined_df.groupby(["source_language", "target_language"]).size())


if __name__ == "__main__":
    combine_training_data(project_name="model_rotation_ref")
