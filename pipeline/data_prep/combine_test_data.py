import pandas as pd
from pathlib import Path


def combine_test_data(project_name):
    # Path to the directory containing the training files
    data_dir = Path(f"outputs/transformed_df/{project_name}")

    # List to store all training DataFrames
    test_dfs = []

    # Read all training CSV files
    for test_file in data_dir.glob("xcomet_*_test.csv"):
        print(f"Reading {test_file}")
        df = pd.read_csv(test_file)
        test_dfs.append(df)

    # Combine all DataFrames
    combined_df = pd.concat(test_dfs, ignore_index=True)

    # Save the combined DataFrame
    output_path = data_dir / "combined_test_data.csv"
    combined_df.to_csv(output_path, index=False)
    print(f"\nCombined test data saved to: {output_path}")
    print(f"Total number of samples: {len(combined_df)}")
    print("\nLanguage pairs in the combined dataset:")
    print(combined_df.groupby(["source_language", "target_language"]).size())


if __name__ == "__main__":
    # combine_test_data()

    project_name = "model_rotation"
    combine_test_data(project_name)
    # project_path = Path(f"outputs/transformed_df/{project_name}")
    # test_file = project_path / "combined_test_data.csv"
    # train_df = pd.read_csv(train_file)
    # test_df = pd.read_csv(test_file)

    # train_ids = train_df["id"].str.split("_").str[1].unique()
    # test_ids = test_df["id"].str.split("_").str[1].unique()
    # print(f"Number of unique ids in combined test data: {len(test_ids)}")
    # print(f"Number of unique ids in combined training data: {len(train_ids)}")
    # # sort the unique ids, convert to int, and save to two files
    # train_ids = sorted(map(int, train_ids))
    # test_ids = sorted(map(int, test_ids))
    # with open("train_ids_hd.txt", "w") as f:
    #     for id in train_ids:
    #         f.write(f"{id}\n")
    # with open("test_ids_hd.txt", "w") as f:
    #     for id in test_ids:
    #         f.write(f"{id}\n")
