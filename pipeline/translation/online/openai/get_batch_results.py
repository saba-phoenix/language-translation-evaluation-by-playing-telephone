from openai import OpenAI
from config import KEY, BATCH_FILE_ID

client = OpenAI(api_key=KEY)

iteration = 16
file_response = client.files.content("file-6q1F9YBetX1bdGE8N2KhGD")
# print(file_response.text)

# save this dictionary to a jsonl file
with open(f"outputs/batch_output/it_{iteration}.jsonl", "w") as f:
    for line in file_response.text.splitlines():
        f.write(line + "\n")
