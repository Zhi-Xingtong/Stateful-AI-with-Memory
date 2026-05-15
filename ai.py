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
ALL_CASES_OPTION = str(len(CASE_OPTIONS) + 1)

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

def _load_experiment_cases():
    options_text = "\n".join([f"{key}. {label}" for key, (label, _) in CASE_OPTIONS.items()])
    while True:
        case = input(f"mode: Experiment\ncase:\n{options_text}\n{ALL_CASES_OPTION}. all_cases\nchoose a number: ")
        if case in CASE_OPTIONS or case == ALL_CASES_OPTION:
            break
        print("Invalid input. Please enter a valid case number.")

    if case == ALL_CASES_OPTION:
        return list(CASE_OPTIONS.values())
    return [CASE_OPTIONS[case]]

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

def _record_path_for_stage(base_record_path, stage):
    if stage == 0:
        return base_record_path
    return _numbered_record_path(base_record_path, stage)

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

def _load_experiment_case_content(path):
    try:
        with open(path, "r") as f:
            content = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("Path not true")
    except json.JSONDecodeError:
        raise ValueError(f"{path} is corrupted")
    ValidCase(content)
    return content

def _resolve_experiment_record_path(base_record_path, turns, create_new_run_after_completion=True):
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
            if create_new_run_after_completion:
                return record_path, [], False
            return None, None, None
        if len(records) < len(turns):
            return record_path, records, True
        index += 1

def _initialize_experiment_case(path, create_new_run_after_completion=True):
    memory = Memory("Exp")

    content = _load_experiment_case_content(path)

    base_record_path = path.replace(".json", "_ans.json")
    record_path, records, reuse_messages = _resolve_experiment_record_path(
        base_record_path,
        content["turns"],
        create_new_run_after_completion=create_new_run_after_completion,
    )

    if record_path is None:
        return None, content["turns"], None, None, None

    if not reuse_messages:
        memory.CaseToMemory(content)

    return memory, content["turns"], record_path, records, reuse_messages

def _resolve_batch_experiment_cases(cases):
    prepared_cases = []
    for case_name, case_path in cases:
        content = _load_experiment_case_content(case_path)
        prepared_cases.append({
            "case_name": case_name,
            "case_path": case_path,
            "content": content,
            "base_record_path": case_path.replace(".json", "_ans.json"),
        })

    stage = 0
    while True:
        stage_states = []
        all_complete = True

        for prepared_case in prepared_cases:
            turns = prepared_case["content"]["turns"]
            record_path = _record_path_for_stage(prepared_case["base_record_path"], stage)
            records = _load_records_if_exists(record_path, turns)

            if records is None:
                stage_states.append(("missing", record_path, []))
                all_complete = False
            elif len(records) < len(turns):
                stage_states.append(("incomplete", record_path, records))
                all_complete = False
            else:
                stage_states.append(("complete", record_path, records))

        if all_complete:
            stage += 1
            continue

        resolved_cases = []
        for prepared_case, (state, record_path, records) in zip(prepared_cases, stage_states):
            if state == "complete":
                resolved_cases.append({
                    "case_name": prepared_case["case_name"],
                    "case_path": prepared_case["case_path"],
                    "skip": True,
                })
                continue

            reuse_messages = state == "incomplete"
            resolved_cases.append({
                "case_name": prepared_case["case_name"],
                "case_path": prepared_case["case_path"],
                "content": prepared_case["content"],
                "turns": prepared_case["content"]["turns"],
                "record_path": record_path,
                "records": records,
                "reuse_messages": reuse_messages,
                "skip": False,
            })

        return resolved_cases

def _initialize_normal_mode(client):
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

    return memory

def _initialize_messages(memory, reuse_messages, messages_path):
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

    return messages

def _run_experiment_case(client, case_name, case_path, case_index, case_total, create_new_run_after_completion=True):
    memory, turns, record_path, records, reuse_messages = _initialize_experiment_case(
        case_path,
        create_new_run_after_completion=create_new_run_after_completion,
    )
    if memory is None:
        print(f"\n[case {case_index}/{case_total}] {case_name} | already complete, skipped")
        return False

    print(f"\n[case {case_index}/{case_total}] {case_name}")
    messages_path = case_path.replace(".json", "_messages.json")
    messages = _initialize_messages(memory, reuse_messages, messages_path)

    turn_index = len(records)
    total_turns = len(turns)

    try:
        while True:
            if turn_index >= len(turns):
                break
            user_input = turns[turn_index]
            turn_index += 1
            _render_progress(turn_index, total_turns, "sending user turn")

            if(len(messages) > 20):
                del messages[1:3]
                with open(messages_path, "w") as f:
                    json.dump(messages, f)

            messages.append({"role": "user", "content": user_input})

            if "I am" in user_input:
                memory.add("user_state", "facts", user_input)

            _render_progress(turn_index, total_turns, "waiting for assistant reply")
            response = _create_chat_completion(
                client,
                model="gpt-5-nano",
                messages=messages
            )

            reply = response.choices[0].message.content
            messages.append({"role": "assistant", "content": reply})

            _render_progress(turn_index, total_turns, "updating memory")
            new_mem = change(user_input, reply)

            if new_mem[0] != "none":
                memory.add("agent_state", new_mem[0], new_mem[1])

            system_prompt = Build(memory.get())
            messages[0] = {"role": "system", "content": system_prompt}

            records.append({"User": user_input, "AI": reply, "new_memory": (new_mem[0], new_mem[1])})
            _render_progress(turn_index, total_turns, "turn complete")
            if turn_index % 10 == 0:
                with open(record_path, "w") as f:
                    json.dump(records, f)
    finally:
        _finish_progress()
        with open(record_path, "w") as f:
            json.dump(records, f)
    return True

def _run_prepared_experiment_case(client, prepared_case, case_index, case_total):
    if prepared_case["skip"]:
        print(f"\n[case {case_index}/{case_total}] {prepared_case['case_name']} | already complete, skipped")
        return False

    case_name = prepared_case["case_name"]
    case_path = prepared_case["case_path"]
    memory = Memory("Exp")
    if not prepared_case["reuse_messages"]:
        memory.CaseToMemory(prepared_case["content"])
    turns = prepared_case["turns"]
    record_path = prepared_case["record_path"]
    records = prepared_case["records"]
    reuse_messages = prepared_case["reuse_messages"]

    print(f"\n[case {case_index}/{case_total}] {case_name}")
    messages_path = case_path.replace(".json", "_messages.json")
    messages = _initialize_messages(memory, reuse_messages, messages_path)

    turn_index = len(records)
    total_turns = len(turns)

    try:
        while True:
            if turn_index >= len(turns):
                break
            user_input = turns[turn_index]
            turn_index += 1
            _render_progress(turn_index, total_turns, "sending user turn")

            if(len(messages) > 20):
                del messages[1:3]
                with open(messages_path, "w") as f:
                    json.dump(messages, f)

            messages.append({"role": "user", "content": user_input})

            if "I am" in user_input:
                memory.add("user_state", "facts", user_input)

            _render_progress(turn_index, total_turns, "waiting for assistant reply")
            response = _create_chat_completion(
                client,
                model="gpt-5-nano",
                messages=messages
            )

            reply = response.choices[0].message.content
            messages.append({"role": "assistant", "content": reply})

            _render_progress(turn_index, total_turns, "updating memory")
            new_mem = change(user_input, reply)

            if new_mem[0] != "none":
                memory.add("agent_state", new_mem[0], new_mem[1])

            system_prompt = Build(memory.get())
            messages[0] = {"role": "system", "content": system_prompt}

            records.append({"User": user_input, "AI": reply, "new_memory": (new_mem[0], new_mem[1])})
            _render_progress(turn_index, total_turns, "turn complete")
            if turn_index % 10 == 0:
                with open(record_path, "w") as f:
                    json.dump(records, f)
    finally:
        _finish_progress()
        with open(record_path, "w") as f:
            json.dump(records, f)
    return True

def main():

    mode = _select_mode()

    load_dotenv()

    api_key=os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing")

    client = OpenAI(api_key = api_key, base_url = "https://api.zhizengzeng.com/v1/")

    if mode == "1":
        cases = _load_experiment_cases()
        total_cases = len(cases)
        if total_cases == 1:
            case_name, case_path = cases[0]
            _run_experiment_case(
                client,
                case_name,
                case_path,
                1,
                1,
                create_new_run_after_completion=True,
            )
            return

        prepared_cases = _resolve_batch_experiment_cases(cases)
        for case_index, prepared_case in enumerate(prepared_cases, start=1):
            _run_prepared_experiment_case(client, prepared_case, case_index, total_cases)
        return

    memory = _initialize_normal_mode(client)
    messages_path = "case_studies/last_turn_messages"
    messages = _initialize_messages(memory, True, messages_path)

    while True:
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

        response = _create_chat_completion(
            client,
            model="gpt-5-nano",
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

if __name__ == "__main__":
    main()
