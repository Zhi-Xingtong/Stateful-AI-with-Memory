from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model = "gpt-5-nano",
    messages = [
        {"role": "system", "content": "You are a friendly AI"}, 
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)