"""Versioned answer-boundary diagnostics. No model or reference-dependent parsing."""
import re

from math_project.data import canonical_number, extract_final_answer

LABEL = re.compile(r"^[ \t]*(?:###[ \t]*)?Final[ \t]+answer[ \t]*:[ \t]*([^\r\n]*)", re.MULTILINE | re.IGNORECASE)
NEXT_QUESTION = re.compile(r"(?<!\S)(?:###[ \t]*)?Question:", re.IGNORECASE)


def boundary(text):
    """Return the first completed boundary; an unfinished numeric prefix is pending."""
    events = []
    label = LABEL.search(text)
    if label:
        end = label.end()
        if text[end:end+2] == "\r\n":
            events.append({"reason": "final_answer_line", "cut": end+2, "observed_at": end+2})
        elif text[end:end+1] == "\n":
            events.append({"reason": "final_answer_line", "cut": end+1, "observed_at": end+1})
    question = NEXT_QUESTION.search(text)
    if question:
        events.append({"reason": "next_question", "cut": question.start(), "observed_at": question.end()})
    return min(events, key=lambda event: event["observed_at"]) if events else None


def normalize_label(value, protocol):
    """Normalize one isolated label value; never look at the reference answer."""
    value = value.strip()
    try:
        return canonical_number(value)
    except ValueError:
        pass
    # Defined wrapper order, a maximum of four layers, no arbitrary prose stripping.
    if value.endswith("."):
        value = value[:-1].strip()
    for _ in range(4):
        previous = value
        for left, right in [("**", "**"), ("`", "`"), ("\\(", "\\)"), ("$", "$")]:
            if value.startswith(left) and value.endswith(right) and len(value) > len(left)+len(right):
                value = value[len(left):-len(right)].strip()
                break
        boxed = re.fullmatch(r"\\boxed\{([^{}]+)\}", value)
        if boxed:
            value = boxed.group(1).strip()
        if value == previous:
            break
    if value and value[0] in protocol["normalization"]["allowed_currency_prefixes"]:
        value = value[1:].strip()
    units = protocol["normalization"]["allowed_units"]
    unit_pattern = "|".join(re.escape(unit) for unit in sorted(units, key=len, reverse=True))
    unit_match = re.fullmatch(rf"(.+?)[ \t]+(?:{unit_pattern})", value, re.IGNORECASE)
    if unit_match:
        value = unit_match.group(1).strip()
    # Spaces surrounding a fraction slash are accepted, not spaces inside integers.
    value = re.sub(r"[ \t]*/[ \t]*", "/", value)
    try:
        return canonical_number(value)
    except ValueError:
        return None


def parse_target(text, finish_reason, protocol):
    """Read only the first target label, never a later invented problem's answer."""
    event = boundary(text)
    target = text[:event["cut"]] if event else text
    reason = event["reason"] if event else finish_reason
    label = LABEL.search(target)
    if label is None:
        return {"answer": None, "strict_numeric_format": False, "parse_status": "missing_final_label"}
    value = label.group(1).strip()
    # A bare numeric prefix at a length cutoff may have lost additional digits.
    if reason == "length" and not target[label.end():].startswith(("\n", "\r\n")):
        return {"answer": None, "strict_numeric_format": False, "parse_status": "incomplete_final_line"}
    strict = True
    try:
        canonical_number(value)
    except ValueError:
        strict = False
    answer = normalize_label(value, protocol)
    return {"answer": answer, "strict_numeric_format": strict,
            "parse_status": "parsed" if answer is not None else "unparseable_final_label"}


def score_target(text, finish_reason, reference, protocol):
    parsed = parse_target(text, finish_reason, protocol)
    return {**parsed, "correct": parsed["answer"] == reference,
            "v1_strict_answer_on_same_text": extract_final_answer(text)}


def repetition_flags(text):
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    counts = {line: lines.count(line) for line in set(lines)}
    return {"repeated_nonempty_line_3plus": max(counts.values(), default=0) >= 3,
            "empty_step_lines_10plus": len(re.findall(r"^\s*Step \d+:\s*$", text, re.MULTILINE)) >= 10}


def validate_protocol(protocol):
    count = protocol["validation_questions"]
    if not 1 <= count <= min(protocol["max_validation_questions"], 20):
        raise ValueError("Diagnostic is capped at 20 validation questions")
    if protocol["scope"] != "validation_only_development_diagnostic_not_final_benchmark":
        raise ValueError("Only validation diagnostics are supported")
    generation = protocol["generation"]
    if generation["do_sample"] or generation["num_beams"] != 1 or not 1 <= generation["max_new_tokens"] <= 512:
        raise ValueError("Diagnostic requires greedy generation with a maximum of 512 new tokens")
    if protocol["modes"] != ["zero_shot", "few_shot"] or protocol["batch_size"] < 1:
        raise ValueError("Expected both frozen prompt modes and a positive batch size")
