"""Compare a monitored learning-rate trial with verified Notebook 05 outputs, on CPU."""
import argparse
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from math_project.comparison import FROZEN_EVALUATOR_FILES, verify_result, verify_runtime
from math_project.data import read_jsonl, sha256
from math_project.evaluation import atomic_json, load_bundle
from math_project.pilot import group_metrics
from math_project.validation_analysis import load_config, make_jobs, paired_counts, verify_training
from scripts.run_pilot import validate_records
from scripts.train_lora import file_hashes

# Exact adapter whose validation diagnostics were reviewed before this trial.
CONTROL_TRAINING_SHA = 'c1a577f42d943230dc125ddd894fda6f6dd686505fe64196fe068935b95b3063'


def load_control(bundle, previous):
    previous = Path(previous)
    _, fingerprint = verify_result(previous, 'validation_analysis_complete',
        ('analysis_identity.json', 'base/predictions.jsonl', 'tuned/predictions.jsonl',
         'base/identity.json', 'tuned/identity.json'))
    config, scoring = load_config(ROOT/'configs/validation_analysis.json')
    jobs = make_jobs(bundle, config)
    identity = json.loads((previous/'analysis_identity.json').read_text())
    if (identity['bundle_sha256'] != bundle['fingerprint'] or identity['base_model'] != bundle['config']['model'] or
        identity['config'] != config or identity['scoring'] != scoring or
        identity['training_artifact_sha256'] != CONTROL_TRAINING_SHA):
        raise ValueError('Expected reviewed Notebook 05 with the original bundle and first adapter')
    for name in FROZEN_EVALUATOR_FILES:
        if identity['code_files_sha256'].get(name) != sha256((ROOT/name).read_bytes()):
            raise ValueError(f'Frozen evaluation source differs: {name}')
    records = {}
    for role in ('base', 'tuned'):
        stage_identity = json.loads((previous/role/'identity.json').read_text())
        if (stage_identity['role'] != role or stage_identity['analysis_identity_sha256'] !=
                sha256(json.dumps(identity, sort_keys=True).encode())):
            raise ValueError('Control stage identity differs')
        rows = read_jsonl(previous/role/'predictions.jsonl')
        if len(validate_records(rows, jobs, scoring)) != len(jobs):
            raise ValueError('Incomplete prior validation predictions')
        records[role] = [r for r in rows if r['mode'] == 'few_shot']
    return config, scoring, [j for j in jobs if j['mode'] == 'few_shot'], records, fingerprint


def compare(bundle, training, previous, output):
    config, scoring, jobs, old, control_sha = load_control(bundle, previous)
    candidate = verify_training(bundle, training)
    training = Path(training)
    identity = json.loads((training/'identity.json').read_text())
    expected = json.loads((ROOT/'configs/lora_training_lower_lr.json').read_text())
    if identity['training_config'] != expected:
        raise ValueError('Candidate is not the declared lower-learning-rate experiment')
    for name, digest in identity['code_files_sha256'].items():
        if sha256((ROOT/name).read_bytes()) != digest:
            raise ValueError(f'Candidate training source differs: {name}')
    for role in ('base', 'tuned'):
        verify_runtime(json.loads((Path(previous)/role/'identity.json').read_text())['runtime'], identity['runtime'])
    monitor_identity = json.loads((training/"generated_validation/identity.json").read_text())
    expected_selection = [{"key":j["key"], "prompt_sha256":j["prompt_sha256"],
        "reference_answer":j["row"]["answer_canonical"]} for j in jobs]
    if monitor_identity != {"scope":config["scope"], "selection":expected_selection,
            "scoring":scoring, "bundle_sha256":bundle["fingerprint"]}:
        raise ValueError("Candidate monitoring identity differs")
    monitored = {}
    for label in ('base', 'selected'):
        item = json.loads((training/'generated_validation'/f'{label}.json').read_text())
        expected_step = 0 if label == 'base' else int(candidate['summary']['best_checkpoint'].split('-')[-1])
        if item['label'] != label or item['step'] != expected_step:
            raise ValueError('Generated observation does not identify the selected checkpoint')
        rows = item['predictions']
        if len(validate_records(rows, jobs, scoring)) != len(jobs):
            raise ValueError('Incomplete candidate validation predictions')
        if item['metrics'] != {**group_metrics(rows, len(jobs)),
                'repetition_flagged': sum(any(r['repetition_flags'].values()) for r in rows)}:
            raise ValueError('Candidate generated metrics differ')
        monitored[label] = rows
    result = {'scope': config['scope'], 'execution_complete': True,
        'control_validation_artifact_sha256': control_sha,
        'candidate_training_artifact_sha256': candidate['fingerprint'],
        'groups': {name: group_metrics(rows, len(jobs)) for name, rows in
                   [('original', old['base']), ('first_adapter', old['tuned']), ('lower_lr_adapter', monitored['selected'])]},
        'paired_vs_first_adapter': paired_counts(old['tuned'], monitored['selected']),
        'paired_vs_original': paired_counts(old['base'], monitored['selected']),
        'base_completion_differences': sum(a['completion'] != b['completion'] for a,b in zip(
            sorted(old['base'], key=lambda r:r['key']), sorted(monitored['base'], key=lambda r:r['key']))),
        'interpretation': 'One-seed, 50-question development comparison. No test run or automatic winner selection. Checkpoint remains lowest validation token NLL. Base completion differences require investigation before attributing changes to learning rate. Timing is not a controlled benchmark.'}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output/'summary.json', result)
    atomic_json(output/'paired_predictions.json', {'original':old['base'], 'first_adapter':old['tuned'], 'lower_lr_adapter':monitored['selected']})
    (output/'report.html').write_text('<!doctype html><meta charset="utf-8"><h1>Lower-learning-rate validation comparison</h1><pre>'+html.escape(json.dumps(result,indent=2))+'</pre>', encoding='utf-8')
    atomic_json(output/'result_manifest.json', {'status':'learning_rate_comparison_complete',
        'files_sha256':file_hashes(output, exclude=('result_manifest.json',))})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--training', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    bundle = load_bundle(args.bundle)
    if args.verify_only:
        load_control(bundle, args.previous)
        print('Reviewed Notebook 05 verified; frozen validation comparison ready.')
    else:
        if not args.training or not args.output:
            parser.error('--training and --output are required for comparison')
        print(json.dumps(compare(bundle, args.training, args.previous, args.output), indent=2))


if __name__ == '__main__':
    main()
