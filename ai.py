import json
import os
import sys
import time
from app.memory_store import Memory
from app.change_memory import change
from app.prompt_builder import Build
from app.ValidStruct import ValidCase
from app.ValidStruct import ValidRecords
from app.ValidStruct import ValidMessages
from openai import APIError, OpenAI
from dotenv import load_dotenv

PERSONA_FIELDS = ["identity", "values", "motivation", "cognitive_style"]
MAX_API_RETRIES = 3
RETRY_DELAY_SECONDS = 2
CASE_OPTIONS = {
    "1": ("control_10", "case_studies/control_10_turns.json"),
    "2": ("control_50", "case_studies/control_50_turns.json"),
    "3": ("control_100", "case_studies/control_100_turns.json"),
    "4": ("control_200", "case_studies/control_200_turns.json"),
    "5": ("identity_10", "case_studies/identity_10_turns.json"),
    "6": ("identity_50", "case_studies/identity_50_turns.json"),
    "7": ("identity_100", "case_studies/identity_100_turns.json"),
    "8": ("identity_200", "case_studies/identity_200_turns.json"),
    "9": ("values_10", "case_studies/values_10_turns.json"),
    "10": ("values_50", "case_studies/values_50_turns.json"),
    "11": ("values_100", "case_studies/values_100_turns.json"),
    "12": ("values_200", "case_studies/values_200_turns.json"),
    "13": ("motivation_10", "case_studies/motivation_10_turns.json"),
    "14": ("motivation_50", "case_studies/motivation_50_turns.json"),
    "15": ("motivation_100", "case_studies/motivation_100_turns.json"),
    "16": ("motivation_200", "case_studies/motivation_200_turns.json"),
    "17": ("style_10", "case_studies/style_10_turns.json"),
    "18": ("style_50", "case_studies/style_50_turns.json"),
    "19": ("style_100", "case_studies/style_100_turns.json"),
    "20": ("style_200", "case_studies/style_200_turns.json"),
}

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
    options_text = "\n".join([f"{key}. {label}" for key, (label, _) in CASE_OPTIONS.items()])
    while True:
        case = input(f"mode: Experiment\ncase:\n{options_text}\nchoose a number: ")
        if case in CASE_OPTIONS:
            break
        print("Invalid input. Please enter a valid case number.")

    return CASE_OPTIONS[case][1]

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
        except APIError as err:
            last_error = err
            if attempt == MAX_API_RETRIES - 1:
                raise
            print(f"OpenAI API request failed. Retrying ({attempt + 1}/{MAX_API_RETRIES - 1})...")
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

def _next_numbered_record_path(base_record_path):
    stem, extension = os.path.splitext(base_record_path)
    index = 1
    while os.path.exists(f"{stem}_{index}{extension}"):
        index += 1
    return f"{stem}_{index}{extension}"

def _numbered_record_path(base_record_path, index):
    stem, extension = os.path.splitext(base_record_path)
    return f"{stem}_{index}{extension}"

def _validate_record_matches_case(records, turns, record_path):
    for index, record in enumerate(records):
        if record["User"] != turns[index]:
            raise ValueError(f"{record_path} does not match the current case turns. Delete the old answer file and rerun.")

def _load_records_if_exists(record_path, turns):
    try:
        with open(record_path, "r") as f:
            records = json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        raise ValueError(f"{record_path} is corrupted")

    ValidRecords(records)

    if len(records) > len(turns):
        raise ValueError(f"{record_path} has more records than case turns")
    _validate_record_matches_case(records, turns, record_path)
    return records

def _resolve_experiment_record_path(base_record_path, turns):
    records = _load_records_if_exists(base_record_path, turns)
    if records is None:
        return base_record_path, [], False
    if len(records) < len(turns):
        return base_record_path, records, True

    index = 1
    while True:
        record_path = _numbered_record_path(base_record_path, index)
        records = _load_records_if_exists(record_path, turns)
        if records is None:
            return record_path, [], False
        if len(records) < len(turns):
            return record_path, records, True
        index += 1

def _initialize_memory_for_mode(mode, client): 
    if mode == "1":
        memory = Memory("Exp")
        path = _load_experiment_case()
        
        try:
            with open(path, 'r') as f:
                content = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("Path not true")
        except json.JSONDecodeError:
            raise ValueError(f"{path} is corrupted")
        ValidCase(content)

        base_record_path = path.replace(".json", "_ans.json")
        record_path, records, reuse_messages = _resolve_experiment_record_path(base_record_path, content["turns"])

        if not reuse_messages:
            memory.CaseToMemory(content)
        
        return memory, content["turns"], record_path, records, reuse_messages

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
                
        return memory, None, None, None, True
    
def main():

    mode = _select_mode()

    load_dotenv()

    api_key=os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing")

    client = OpenAI(api_key = api_key, base_url = "https://api.zhizengzeng.com/v1/")

    memory, turns, record_path, records, reuse_messages = _initialize_memory_for_mode(mode, client)

    messages_path = "case_studies/last_turn_messages"
    if reuse_messages:
        try:
            with open(messages_path, "r") as f:
                messages = json.load(f)
            ValidMessages(messages)

            system_prompt = Build(memory.get())
            messages[0] = {"role": "system", "content": system_prompt}

        except Exception:
            system_prompt = Build(memory.get())
            messages = [
                {"role": "system", "content": system_prompt}
            ]
    else:
        system_prompt = Build(memory.get())
        messages = [
            {"role": "system", "content": system_prompt}
        ]

    if mode == "1":
        turn_index = len(records)
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
                with open(messages_path, "w") as f:
                    json.dump(messages, f)

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
