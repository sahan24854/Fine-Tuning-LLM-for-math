"""Dataset preparation and protocol helpers; standard-library dependencies only."""

import hashlib
import io
import json
import random
import re
import unicodedata
import urllib.request
from collections import Counter
from fractions import Fraction
from pathlib import Path


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _jsonl_records(lines, source):
    records = []
    for line_number, line in enumerate(lines, 1):
        try:
            records.append(json.loads(line))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError(f"Invalid JSONL in {source}, physical line {line_number}: {exc}") from exc
    return records


def read_jsonl(path):
    """Read LF-delimited records without splitting valid Unicode inside strings."""
    with Path(path).open("rb") as file:
        return _jsonl_records(file, str(path))


def question_key(question):
    # Conservative exact-match normalization; not a semantic decontamination claim.
    return " ".join(unicodedata.normalize("NFKC", question).casefold().split())


def canonical_number(value):
    """Return an exact rational string. Reject units, prose, bad grouping and NaN."""
    text = str(value).strip().replace("−", "-")
    integer = r"(?:\d{1,3}(?:,\d{3})+|\d+)"
    number = rf"[+-]?(?:{integer}(?:\.\d+)?|\.\d+)"
    if not re.fullmatch(rf"{number}(?:/[+-]?\d+)?", text):
        raise ValueError(f"Unsupported numeric answer: {value!r}")
    try:
        return str(Fraction(text.replace(",", "")))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"Invalid numeric answer: {value!r}") from exc


def extract_final_answer(completion):
    """Score generated completion only, never prompt examples or arbitrary last numbers."""
    lines = re.findall(r"^[ \t]*Final answer:[ \t]*([^\n]+)$", completion, flags=re.MULTILINE | re.IGNORECASE)
    if len(lines) != 1:
        return None
    try:
        return canonical_number(lines[0])
    except ValueError:
        return None


def download_source(source, cache_dir):
    """Immutable source bytes verified before parsing, including cached bytes."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / source["sha256"]
    if path.exists():
        data = path.read_bytes()
    else:
        request = urllib.request.Request(source["url"], headers={"User-Agent": "math-transfer-learning/1"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
    if sha256(data) != source["sha256"]:
        raise ValueError(f"Checksum mismatch for {source['url']}; refusing this source")
    if not path.exists():
        path.write_bytes(data)
    return data


def parse_gsm8k(data, split):
    rows = []
    for index, source in enumerate(_jsonl_records(io.BytesIO(data), "GSM8K source")):
        solution, answer = source["answer"].rsplit("####", 1)
        # Remove calculator annotations, retaining the human-readable calculations.
        solution = re.sub(r"<<.*?>>", "", solution).strip()
        rows.append({"id": f"gsm8k/{split}/{index:05d}", "dataset": "gsm8k",
                     "question": source["question"].strip(), "solution": solution,
                     "answer": answer.strip(), "answer_canonical": canonical_number(answer)})
    return rows


def parse_svamp(data):
    return [{"id": f"svamp/{source['ID']}", "dataset": "svamp",
             "question": f"{source['Body'].strip()} {source['Question'].strip()}",
             "answer": str(source["Answer"]),
             "answer_canonical": canonical_number(source["Answer"]), "category": source["Type"]}
            for source in json.loads(data)]


def split_training(rows, held_out, seed, validation_fraction):
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    forbidden = {question_key(row["question"]) for row in held_out}
    unique = {}
    duplicates = overlap = 0
    for row in rows:
        key = question_key(row["question"])
        if key in forbidden:
            overlap += 1
            continue
        if key in unique:
            if row["answer_canonical"] != unique[key]["answer_canonical"]:
                raise ValueError(f"Conflicting labels for duplicate training question {row['id']}")
            duplicates += 1
        else:
            unique[key] = row
    # Hash ranking is independent of input ordering and Python RNG implementation.
    ranked = sorted(unique.values(), key=lambda row: sha256(f"{seed}:{question_key(row['question'])}".encode()))
    count = round(len(ranked) * validation_fraction)
    if count < 1 or count >= len(ranked):
        raise ValueError("Not enough unique training examples for this split")
    return ranked[count:], ranked[:count], {"training_duplicates_removed": duplicates,
                                          "training_test_overlap_removed": overlap}


def arithmetic_set(seed, per_category):
    """Fresh evaluation-only examples with exact, independently checkable labels."""
    if not 1 <= per_category <= 100:
        raise ValueError("arithmetic_per_category must be between 1 and 100")
    rng = random.Random(seed)
    rows, seen = [], set()
    for number_range, bounds in [("small", (1, 20)), ("large", (100, 999))]:
        for operation in ["add", "subtract", "multiply", "divide", "two_step"]:
            for index in range(per_category):
                while True:
                    a, b = rng.randint(*bounds), rng.randint(*bounds)
                    c = rng.randint(1, min(a * b, 100))
                    if operation == "add":
                        question, answer = f"What is {a} + {b}?", a + b
                    elif operation == "subtract":
                        a, b = max(a, b), min(a, b)
                        question, answer = f"What is {a} - {b}?", a - b
                    elif operation == "multiply":
                        question, answer = f"What is {a} * {b}?", a * b
                    elif operation == "divide":
                        question, answer = f"What is {a * b} / {b}?", a
                    else:
                        question = f"A shop has {a} boxes with {b} pencils each. It sells {c} pencils. How many pencils remain?"
                        answer = a * b - c
                    if question not in seen:
                        seen.add(question)
                        break
                rows.append({"id": f"arithmetic/{number_range}/{operation}/{index:03d}",
                             "dataset": "arithmetic", "question": question,
                             "answer": str(answer), "answer_canonical": str(answer),
                             "operation": operation, "number_range": number_range,
                             "operands": [a, b, c] if operation == "two_step" else [a, b]})
    return rows


def prepare_data(config, cache_dir):
    raw = {name: download_source(source, cache_dir) for name, source in config["sources"].items()}
    training = parse_gsm8k(raw["gsm8k_train"], "train")
    test = parse_gsm8k(raw["gsm8k_test"], "test")
    svamp = parse_svamp(raw["svamp_test"])
    arithmetic = arithmetic_set(config["seed"], config["arithmetic_per_category"])
    train, validation, audit = split_training(training, test + svamp + arithmetic,
                                             config["seed"], config["validation_fraction"])
    splits = {"train": train, "validation": validation, "gsm8k_test": test,
              "svamp_test": svamp, "arithmetic_test": arithmetic}
    audit["source_counts"] = {"gsm8k_train": len(training), "gsm8k_test": len(test), "svamp_test": len(svamp)}
    audit["counts_before_length_filter"] = {name: len(rows) for name, rows in splits.items()}
    audit["duplicate_questions_within_evaluation"] = {
        name: len(rows) - len({question_key(row["question"]) for row in rows})
        for name, rows in splits.items() if name.endswith("_test")}
    audit["gsm8k_svamp_exact_overlap"] = len({question_key(r["question"]) for r in test} &
                                             {question_key(r["question"]) for r in svamp})
    audit["limits"] = ["Exact normalized question checks do not detect semantic paraphrases.",
                       "Pretraining contamination is unknown; large numbers are a stress test, not proven out-of-distribution.",
                       "Test sets are preserved; evaluate benchmarks separately, not as one pooled accuracy."]
    for name, rows in splits.items():
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate IDs in {name}")
        if any(not row["question"] for row in rows):
            raise ValueError(f"Empty question in {name}")
    return splits, audit


def completion(row):
    return f" {row['solution']}\nFinal answer: {row['answer']}"


def prompt(question, demonstrations=()):
    prefix = "Solve the problem. Finish with one line: Final answer: <number>\n\n"
    examples = "".join(f"Question: {row['question']}\nSolution:{completion(row)}\n\n" for row in demonstrations)
    return f"{prefix}{examples}Question: {question}\nSolution:"


def length_summary(values):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot summarize an empty dataset")
    return {"min": ordered[0], "median": ordered[len(ordered) // 2],
            "p95": ordered[int((len(ordered) - 1) * .95)], "max": ordered[-1]}


def tokenize_and_audit(splits, config, tokenizer, audit):
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define an EOS token")
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    kept, excluded, lengths = [], [], []
    for row in splits["train"]:
        # Explicit boundary: later SFT must concatenate these two encoded segments,
        # mask prompt IDs with -100, and append EOS. No silent truncation.
        prompt_ids = encode(prompt(row["question"]))
        completion_ids = encode(completion(row)) + [tokenizer.eos_token_id]
        count = len(prompt_ids) + len(completion_ids)
        lengths.append(count)
        if count <= config["max_training_tokens"]:
            kept.append({**row, "prompt": prompt(row["question"]), "completion": completion(row),
                         "prompt_token_count": len(prompt_ids), "completion_token_count": len(completion_ids)})
        else:
            excluded.append({"id": row["id"], "token_count": count, "reason": "exceeds_max_training_tokens"})
    if not kept:
        raise ValueError("All training rows exceed the sequence limit")
    splits = {**splits, "train": kept}
    candidates = [row for row in kept if len(encode(f"Question: {row['question']}\nSolution:{completion(row)}\n\n"))
                  <= config["demonstration_max_tokens"]]
    demonstrations = candidates[:config["few_shot_count"]]
    if len(demonstrations) != config["few_shot_count"]:
        raise ValueError("Not enough short training demonstrations")
    audit["training_tokens_before_filter"] = length_summary(lengths)
    audit["excluded_training_rows"] = len(excluded)
    audit["evaluation_prompt_tokens"] = {}
    max_prompt = config["context_window_tokens"] - config["generation"]["max_new_tokens"]
    if config["context_window_tokens"] > tokenizer.model_max_length:
        raise ValueError("Configured context exceeds tokenizer model_max_length")
    for name, rows in splits.items():
        if name == "train":
            continue
        audit["evaluation_prompt_tokens"][name] = {}
        for mode, shots in [("zero_shot", []), ("few_shot", demonstrations)]:
            lengths = [len(encode(prompt(row["question"], shots))) for row in rows]
            audit["evaluation_prompt_tokens"][name][mode] = length_summary(lengths)
            if max(lengths) > max_prompt:
                raise ValueError(f"{name}/{mode} exceeds prompt budget {max_prompt}; revise protocol before evaluating")
    audit["final_counts"] = {name: len(rows) for name, rows in splits.items()}
    audit["arithmetic_categories"] = dict(Counter(f"{r['number_range']}/{r['operation']}" for r in splits["arithmetic_test"]))
    return splits, demonstrations, excluded
