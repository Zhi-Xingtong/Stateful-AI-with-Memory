from memory_store import Memory
from change_memory import change
from prompt_builder import Build
from openai import OpenAI
from dotenv import load_dotenv
import os

memory = Memory()

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

user_input = input("You want ai to play: ")

internal_messages = [
    {"role": "system", "content": f"""
You are {user_input} now!

generate your

identity:

values:

motivation:

cognitive_style:

your must use format above to generate your message
"""}
]

internal_response = client.chat.completions.create(
    model = "gpt-5-nano",
    messages=internal_messages
)

for line in internal_response.choices[0].message.content.split("\n"):
    line = line.lower().strip()
    if "identity:" in line:
        memory.add("agent_state", "identity", line.split("identity:")[1])
    if "values:" in line:
        memory.add("agent_state", "values", line.split("values:")[1])
    if "motivation:" in line:
        memory.add("agent_state", "motivation", line.split("motivation:")[1])
    if "cognitive_style:" in line:
        memory.add("agent_state", "cognitive_style", line.split("cognitive_style:")[1])

system_prompt = Build(memory.get())

messages = [
    {"role": "system", "content": system_prompt}
]

while True:

    if(len(messages) > 20): 
        del messages[1:3]

    user_input = input("You: ")

    messages.append({"role": "user", "content": user_input})

    if "I am" in user_input:
        memory.add("user_state", "facts", user_input)

    response = client.chat.completions.create(
        model = "gpt-5-nano",
        messages=messages
    )

    reply = response.choices[0].message.content
    print("AI: ", reply)

    messages.append({"role": "assistant", "content": reply})

    new_mem = change(user_input, reply)

    if new_mem[0] != "none":
        memory.add("agent_state", new_mem[0], new_mem[1])
    
    system_prompt = Build(memory.get())

    messages[0] = {"role": "system", "content": system_prompt}
