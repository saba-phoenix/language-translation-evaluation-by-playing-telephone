import os
import pandas as pd
import torch

torch.set_float32_matmul_precision("high")
from comet import download_model, load_from_checkpoint
from pathlib import Path


def process_xcomet_files(directory_name, model_name):
    """
    Process XCOMET test files with a specified model.

    Args:
        directory_name (str): Name of the directory inside transformed_df
        model_name (str): Model name (without era28/ prefix)

    Returns:
        None
    """

    model_path = download_model(f"era28/{model_name}")
    model = load_from_checkpoint(model_path)

    # Create outputs directory if it doesn't exist
    base_dir = Path("outputs/transformed_df")
    output_dir = base_dir / directory_name / "prediction_test"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Get all XCOMET test files from the specified directory
    data_dir = base_dir / directory_name
    # data_dir.mkdir(exist_ok=True, parents=True)
    test_files = list(data_dir.glob("xcomet_*_test.csv"))

    for test_file in test_files:
        print(f"Processing {test_file.name}...")

        # Read the test file
        df = pd.read_csv(test_file)

        # Prepare data for model
        data = []
        for _, row in df.iterrows():
            data.append({"src": row["src"], "mt": row["mt"]})

        # Get predictions
        predictions = model.predict(data, batch_size=8, gpus=1)

        # Add predictions to dataframe
        df["predicted_score"] = predictions.scores

        # Save to output file
        output_file = output_dir / f"{test_file.name}"
        df.to_csv(output_file, index=False)
        print(f"Saved predictions to {output_file}")

    print("All files processed successfully!")


process_xcomet_files("lowdiv_dir", "qe-language-rotation-ldd")
