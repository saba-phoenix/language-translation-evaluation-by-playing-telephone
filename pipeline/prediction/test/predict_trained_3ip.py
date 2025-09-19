import os
import pandas as pd
import torch
from comet import download_model, load_from_checkpoint
from pathlib import Path

# Set up model
torch.set_float32_matmul_precision("high")
model_path = download_model("era28/regression-model-english")
model = load_from_checkpoint(model_path)

# Create outputs directory if it doesn't exist
output_dir = Path("outputs/transformed_df")
output_dir.mkdir(exist_ok=True)

# Get all XCOMET test files
data_dir = Path("outputs/transformed_df")
test_files = list(data_dir.glob("xcomet_ref_*_test.csv"))

for test_file in test_files:
    print(f"Processing {test_file.name}...")

    # Read the test file
    df = pd.read_csv(test_file)

    # Prepare data for model
    data = []
    for _, row in df.iterrows():
        data.append({"src": row["src"], "mt": row["mt"], "ref": row["ref"]})

    # Get predictions
    predictions = model.predict(data, batch_size=8, gpus=1)

    # Add predictions to dataframe
    df["predicted_score"] = predictions.scores

    # Save to output file
    output_file = output_dir / f"predictions_{test_file.name}"
    df.to_csv(output_file, index=False)
    print(f"Saved predictions to {output_file}")

print("All files processed successfully!")
