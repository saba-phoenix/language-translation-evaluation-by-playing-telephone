from openai import OpenAI
from config import KEY, BATCH_FILE_ID

client = OpenAI(api_key=KEY)

batch_input_file_id = BATCH_FILE_ID
res = client.batches.create(
    input_file_id=batch_input_file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={"description": "third iteration translation batch"},
)
print(res)
