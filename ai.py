import json
import os
import sys
import time
from app.memory_store import Memory
from app.change_memory import change
from app.prompt_builder import Build
from app.ValidStruct import ValidCase
from openai import APITimeoutError, OpenAI
from dotenv import load_dotenv

PERSONA_FIELDS = ["identity", "values", "motivation", "cognitive_style"]
MAX_API_RETRIES = 3
RETRY_DELAY_SECONDS = 2

def _render_progress(current, total, stage):
    if total <= 0:
        return
    bar_width = 24
    filled = int(bar_width * current / total)
    bar = "#" * filled + "-" * (bar_width - filled)
    sys.stdout.write(f"\r[{bar}] {current}/{total} | {stage}")
    sys.stdout.flush()

def _finish_progress():
    sys.stdout.write("\n")
    sys.stdout.flush()

def _select_mode():
    while True:
        mode = input("mode: \n1. Experiment\n2. Normal\nchoose a number: ")
        if mode in {"1", "2"}:
            break
        print("Invalid input. Please enter 1 or 2.")
    
    return mode

def _load_experiment_case():
    while True:
        case = input("mode: Experiment\ncase: \n1. 10_turns\n2. 20_turns\n3. 50_turns\n4. 100_turns\n5. 200_turns\nchoose a number")
        if case in {"1", "2", "3", "4", "5"}:
            break
        print ("Invalid input. Please enter a number between 1-5")

    return case

def _has_nonempty_items(items):
    for item in items:
        if item.strip():
            return True
    return False

def _missing_persona_fields(memory):
    agent_state = memory.get()["agent_state"]
    missing = []
    for field in PERSONA_FIELDS:
        if not _has_nonempty_items(agent_state[field]):
            missing.append(field)
    return missing

def _create_chat_completion(client, model, messages):
    last_error = None
    for attempt in range(MAX_API_RETRIES):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages
            )
        except APITimeoutError as err:
            last_error = err
            if attempt == MAX_API_RETRIES - 1:
                raise
            print(f"OpenAI request timed out. Retrying ({attempt + 1}/{MAX_API_RETRIES - 1})...")
            time.sleep(RETRY_DELAY_SECONDS)
    raise last_error

def _generate_persona_field(client, role, field):
    prompt = f"""
You are generating one stable self-state field for a persistent roleplayed character.

Role:
{role}

Target field:
{field}

Rules:
- Output exactly one sentence.
- Output only the content for the target field.
- Make it suitable for long-term character memory.
- Do not mention prompts, memory, or system instructions.
- Do not mention a specific conversation.
- Use first-person style.
"""

    response = _create_chat_completion(
        client,
        model="gpt-5-nano",
        messages=[{"role": "system", "content": prompt}]
    )

    return response.choices[0].message.content.strip()

def _initialize_memory_for_mode(mode, client): 
    if mode == "1":
        memory = Memory("Exp")
        case = _load_experiment_case()

        if case == "1":
            path = "case_studies/10_turns.json"
        elif case == "2":
            path = "case_studies/20_turns.json"
        elif case == "3":
            path = "case_studies/50_turns.json"
        elif case == "4":
            path = "case_studies/100_turns.json"
        elif case == "5":
            path = "case_studies/200_turns.json"
        
        try:
            with open(path, 'r') as f:
                content = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("Path not true")
        except json.JSONDecodeError:
            raise ValueError(f"{path} is corrupted")
        ValidCase(content)
        
        memory.CaseToMemory(content)
        
        return memory, content["turns"], path.replace(".json", "_ans.json")

    else:
        memory = Memory("Norm")
        agent_state = memory.get()["agent_state"]

        if not agent_state["role"].strip():
            while True:
                user_input = input("You want ai to play: ")
                if user_input.strip():
                    break
            memory.set_role(user_input.strip())

        role = memory.get()["agent_state"]["role"]
        missing_fields = _missing_persona_fields(memory)
        total_missing = len(missing_fields)

        for index, field in enumerate(missing_fields, start=1):
            print(f"Generating missing persona field {index}/{total_missing}: {field}")
            generated = _generate_persona_field(client, role, field)
            if generated:
                memory.add("agent_state", field, generated)
                
        return memory, None, None
    
def main():

    mode = _select_mode()

    load_dotenv()

    api_key=os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing")

    client = OpenAI(api_key = api_key, base_url = "https://api.zhizengzeng.com/v1/")

    memory, turns, record_path = _initialize_memory_for_mode(mode, client)

    system_prompt = Build(memory.get())

    messages = [
        {"role": "system", "content": system_prompt}
    ]

    if mode == "1":
        records = []
        turn_index = 0
        total_turns = len(turns)

    try:
        while True:
            if turns is not None:
                if turn_index >= len(turns):
                    break
                user_input = turns[turn_index]
                turn_index += 1
                _render_progress(turn_index, total_turns, "sending user turn")
            else:
                user_input = input("You: ").strip()
                if user_input.lower() in {"quit"}:
                    break
            
            if(len(messages) > 20): 
                del messages[1:3]

            messages.append({"role": "user", "content": user_input})

            if "I am" in user_input:
                memory.add("user_state", "facts", user_input)

            if turns is not None:
                _render_progress(turn_index, total_turns, "waiting for assistant reply")
            response = _create_chat_completion(
                client,
                model="gpt-5-nano",
                messages=messages
            )

            reply = response.choices[0].message.content
            if mode != "1":
                print("AI: ", reply)

            messages.append({"role": "assistant", "content": reply})

            if turns is not None:
                _render_progress(turn_index, total_turns, "updating memory")
            new_mem = change(user_input, reply)

            if new_mem[0] != "none":
                memory.add("agent_state", new_mem[0], new_mem[1])
            
            system_prompt = Build(memory.get())

            messages[0] = {"role": "system", "content": system_prompt}

            if mode == "1":
                records.append({"User": user_input, "AI": reply, "new_memory": (new_mem[0], new_mem[1])})
                _render_progress(turn_index, total_turns, "turn complete")
                if turn_index % 10 == 0:
                    with open(record_path, "w") as f:
                        json.dump(records, f)
    finally:
        if mode == "1":
            _finish_progress()
            with open(record_path, "w") as f:
                json.dump(records, f)

if __name__ == "__main__":
    main()
