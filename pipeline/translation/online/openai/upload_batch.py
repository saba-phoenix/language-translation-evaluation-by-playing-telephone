from openai import OpenAI
from config import KEY

client = OpenAI(api_key=KEY)

iteration = 16

batch_input_file = client.files.create(
    file=open(f"outputs/batch_request/it_{iteration}.jsonl", "rb"), purpose="batch"
)

print(batch_input_file)

batch_input_file_id = batch_input_file.id
res = client.batches.create(
    input_file_id=batch_input_file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={"description": "16th iteration translation batch"},
)
print(res)
# Save the batch ID for later use
batch_id = res.id
print(f"Batch ID: {batch_id}")
