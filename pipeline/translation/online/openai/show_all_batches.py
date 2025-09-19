from openai import OpenAI
from config import KEY

client = OpenAI(api_key=KEY)


print(client.batches.list(limit=10))
