import pandas as pd

# Read the text file
file_path = "evaluation_results_xcomet.txt"  # Update with your actual file path
df = pd.read_csv(
    file_path, delimiter="|"
)  # Change delimiter based on the file (e.g., ',', '\t', ' ')

# Save to Excel
excel_filename = "output_xcomet.xlsx"
df.to_excel(excel_filename, index=False)

print(f"Excel file '{excel_filename}' has been created successfully!")
