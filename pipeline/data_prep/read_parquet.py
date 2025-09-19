import pandas as pd

# Read parquet file
df = pd.read_parquet("outputs/all_eval/ip_op_lowdiv_relative_ref.parquet")

# Convert back to list of dicts if needed
data = df.to_dict("records")

# Or work directly with DataFrame
print(data[0])
