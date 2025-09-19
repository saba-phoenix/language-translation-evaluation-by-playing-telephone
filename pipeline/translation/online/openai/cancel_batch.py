from openai import OpenAI
from config import KEY, BATCH_FILE_ID

client = OpenAI(api_key=KEY)


client.batches.cancel("batch_67fd3830664c81908b6171b234ae443c")
