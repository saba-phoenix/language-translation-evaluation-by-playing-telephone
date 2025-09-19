from openai import OpenAI
from config import KEY, BATCH_ID

client = OpenAI(
    api_key="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"
)

batch = client.batches.retrieve("batch_682aaa1002e081908b2f32805e211a19")
print(batch)
