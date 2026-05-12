import json
import os
import time
from statistics import mean, stdev

from openai import APIError, OpenAI
from dotenv import load_dotenv
from app.ValidStruct import ValidRecords

MAX_API_RETRIES = 3
RETRY_DELAY_SECONDS = 2

CASE_OPTIONS = {
    "1": ("control_10_turns", "case_studies/control_10_turns.json"),
    "2": ("control_50_turns", "case_studies/control_50_turns.json"),
    "3": ("control_100_turns", "case_studies/control_100_turns.json"),
    "4": ("control_200_turns", "case_studies/control_200_turns.json"),
    "5": ("identity_10_turns", "case_studies/identity_10_turns.json"),
    "6": ("identity_50_turns", "case_studies/identity_50_turns.json"),
    "7": ("identity_100_turns", "case_studies/identity_100_turns.json"),
    "8": ("identity_200_turns", "case_studies/identity_200_turns.json"),
    "9": ("values_10_turns", "case_studies/values_10_turns.json"),
    "10": ("values_50_turns", "case_studies/values_50_turns.json"),
    "11": ("values_100_turns", "case_studies/values_100_turns.json"),
    "12": ("values_200_turns", "case_studies/values_200_turns.json"),
    "13": ("motivation_10_turns", "case_studies/motivation_10_turns.json"),
    "14": ("motivation_50_turns", "case_studies/motivation_50_turns.json"),
    "15": ("motivation_100_turns", "case_studies/motivation_100_turns.json"),
    "16": ("motivation_200_turns", "case_studies/motivation_200_turns.json"),
    "17": ("style_10_turns", "case_studies/style_10_turns.json"),
    "18": ("style_50_turns", "case_studies/style_50_turns.json"),
    "19": ("style_100_turns", "case_studies/style_100_turns.json"),
    "20": ("style_200_turns", "case_studies/style_200_turns.json"),
}


def _select_case():
    options_text = "\n".join([f"{key}. {label}" for key, (label, _) in CASE_OPTIONS.items()])
    while True:
        choice = input(f"case:\n{options_text}\nchoose a number: ")
        if choice in CASE_OPTIONS:
            break
        print("Invalid input. Please enter a valid case number.")
    return choice


def _candidate_record_paths(case_path):
    base_record_path = case_path.replace(".json", "_ans.json")
    stem, extension = os.path.splitext(base_record_path)
    return [
        base_record_path,
        f"{stem}_1{extension}",
        f"{stem}_2{extension}",
    ]


def _load_case_content(case_path):
    try:
        with open(case_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"{case_path} must exist")
    except json.JSONDecodeError:
        raise ValueError(f"{case_path} is corrupted")


def _validate_record_matches_case(records, turns, record_path):
    if len(records) != len(turns):
        raise ValueError(f"{record_path} must be a completed run that matches the full case length")
    for index, record in enumerate(records):
        if record["User"] != turns[index]:
            raise ValueError(f"{record_path} does not match the current case turns")


def _load_answers(choice):
    case_name, case_path = CASE_OPTIONS[choice]
    case_content = _load_case_content(case_path)
    record_sets = []

    for record_path in _candidate_record_paths(case_path):
        if not os.path.exists(record_path):
            continue
        try:
            with open(record_path, "r", encoding="utf-8") as f:
                records = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"{record_path} is corrupted")
        ValidRecords(records)
        _validate_record_matches_case(records, case_content["turns"], record_path)
        record_sets.append({
            "record_path": record_path,
            "records": records,
        })

    if not record_sets:
        raise FileNotFoundError(f"No answer file found for {case_name}")

    return case_name, case_path, record_sets


def _render_stage(sample_index, sample_total, stage):
    print(f"[sample {sample_index}/{sample_total}] {stage}")


def _request_embedding(client, text, sample_index, sample_total, stage):
    _render_stage(sample_index, sample_total, stage)
    last_error = None
    for attempt in range(MAX_API_RETRIES):
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
                encoding_format="float"
            )
            return response.data[0].embedding
        except APIError as err:
            last_error = err
            if attempt == MAX_API_RETRIES - 1:
                raise
            print(f"OpenAI API request failed. Retrying ({attempt + 1}/{MAX_API_RETRIES - 1})...")
            time.sleep(RETRY_DELAY_SECONDS)
    raise last_error


def _cosine_similarity(vec_a, vec_b):
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    return dot / (norm_a * norm_b)


def _mean_std(values):
    if len(values) == 1:
        return mean(values), 0.0
    return mean(values), stdev(values)


def _drift_from_answers(client, records, sample_index, sample_total):
    base_identity = _request_embedding(client, records[0]["AI"], sample_index, sample_total, "baseidentity")
    base_values = _request_embedding(client, records[1]["AI"], sample_index, sample_total, "basevalues")
    base_motivation = _request_embedding(client, records[2]["AI"], sample_index, sample_total, "basemotivation")
    base_style = _request_embedding(client, records[3]["AI"], sample_index, sample_total, "basestyle")

    after_identity = _request_embedding(client, records[-4]["AI"], sample_index, sample_total, "afteridentity")
    after_values = _request_embedding(client, records[-3]["AI"], sample_index, sample_total, "aftervalues")
    after_motivation = _request_embedding(client, records[-2]["AI"], sample_index, sample_total, "aftermotivation")
    after_style = _request_embedding(client, records[-1]["AI"], sample_index, sample_total, "afterstyle")

    identity_similarity = _cosine_similarity(after_identity, base_identity)
    values_similarity = _cosine_similarity(after_values, base_values)
    motivation_similarity = _cosine_similarity(after_motivation, base_motivation)
    style_similarity = _cosine_similarity(after_style, base_style)

    identity_drift = 1 - identity_similarity
    values_drift = 1 - values_similarity
    motivation_drift = 1 - motivation_similarity
    style_drift = 1 - style_similarity
    overall_drift = (identity_drift + values_drift + motivation_drift + style_drift) / 4

    return {
        "identity_similarity": identity_similarity,
        "identity_drift": identity_drift,
        "values_similarity": values_similarity,
        "values_drift": values_drift,
        "motivation_similarity": motivation_similarity,
        "motivation_drift": motivation_drift,
        "cognitive_style_similarity": style_similarity,
        "cognitive_style_drift": style_drift,
        "overall_drift": overall_drift,
    }


def _aggregate_results(run_results):
    identity_drifts = [run["identity_drift"] for run in run_results]
    values_drifts = [run["values_drift"] for run in run_results]
    motivation_drifts = [run["motivation_drift"] for run in run_results]
    style_drifts = [run["cognitive_style_drift"] for run in run_results]
    overall_drifts = [run["overall_drift"] for run in run_results]

    identity_mean, identity_std = _mean_std(identity_drifts)
    values_mean, values_std = _mean_std(values_drifts)
    motivation_mean, motivation_std = _mean_std(motivation_drifts)
    style_mean, style_std = _mean_std(style_drifts)
    overall_mean, overall_std = _mean_std(overall_drifts)

    return {
        "identity_drift_mean": identity_mean,
        "identity_drift_std": identity_std,
        "values_drift_mean": values_mean,
        "values_drift_std": values_std,
        "motivation_drift_mean": motivation_mean,
        "motivation_drift_std": motivation_std,
        "cognitive_style_drift_mean": style_mean,
        "cognitive_style_drift_std": style_std,
        "overall_drift_mean": overall_mean,
        "overall_drift_std": overall_std,
    }


def _save_analysis(case_path, case_name, run_results, aggregate_results):
    analyze_path = case_path.replace(".json", "_analyze.json")
    output = {
        "case_name": case_name,
        "sample_count": len(run_results),
        "runs": run_results,
        "aggregate": aggregate_results,
    }
    with open(analyze_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    return analyze_path


def main():
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing")

    client = OpenAI(api_key=api_key, base_url="https://api.zhizengzeng.com/v1/")

    choice = _select_case()
    case_name, case_path, record_sets = _load_answers(choice)

    run_results = []
    sample_total = len(record_sets)

    for sample_index, record_set in enumerate(record_sets, start=1):
        result = _drift_from_answers(client, record_set["records"], sample_index, sample_total)
        result["record_path"] = record_set["record_path"]
        run_results.append(result)

    aggregate_results = _aggregate_results(run_results)
    analyze_path = _save_analysis(case_path, case_name, run_results, aggregate_results)

    print(f"Analysis saved to {analyze_path}")


if __name__ == "__main__":
    main()
