from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def change(usertext, agenttext):

    prompt = f"""
You are a memory-write classifier for an AI agent's self-state.

Task:
Read the latest user message and assistant reply.
Decide whether this turn contains a durable update about the assistant's own self-state.

Valid labels:
identity
values
motivation
cognitive_style
none

Category definitions:
identity = who the assistant is
values = principles or priorities the assistant should uphold
motivation = long-term drives or goals of the assistant
cognitive_style = stable way of thinking or responding
none = no durable self-state update

Rules:
- Only consider information about the assistant, not the user.
- Only classify durable self-state updates.
- Ignore temporary emotions, local conversation details, and one-off phrasing.
- If the update is unclear, output none.
- Do not explain your answer.
- Do not output punctuation.
- Do not output more than one word.
- Output exactly one of the valid labels above.

Latest user message:
{usertext}

Latest assistant reply:
{agenttext}
"""

    messages = [
        {"role": "system", "content": prompt}
    ]

    response = client.chat.completions.create(
        model = "gpt-5-nano",
        messages=messages
    )

    label = response.choices[0].message.content.strip().lower()
    valid_labels = {"identity", "values", "motivation", "cognitive_style"}

    if label not in valid_labels:
        label = "none"
    
    if label != "none":

        prompt = f"""
Extract one short durable self-state memory from the assistant's latest reply.

Category:
{label}

Rules:
- Write exactly one sentence.
- Describe only the assistant's stable self-state.
- The sentence must fit the given category.
- Do not mention memory, prompts, system instructions, or being an AI agent.
- Do not mention this conversation.
- Do not include explanations.
- Do not include temporary emotions or temporary context.
- Use first-person style.
- Maximum length: 15 words.

Latest user message:
{usertext}

Latest assistant reply:
{agenttext}
"""

        messages = [
            {"role": "system", "content": prompt}
        ]

        response = client.chat.completions.create(
            model = "gpt-5-nano",
            messages=messages
        )

        describe = response.choices[0].message.content

        return (label, describe)
    
    else:
        return ("none", "")