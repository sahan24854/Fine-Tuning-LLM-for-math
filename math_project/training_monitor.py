"""Observational generation checks; never select checkpoints or change training RNG."""
from contextlib import contextmanager
import json
from pathlib import Path
import random

from math_project.data import sha256
from math_project.evaluation import atomic_json
from math_project.inference import ModelRunner
from math_project.inference_v2 import BoundedRunner
from math_project.pilot import group_metrics
from math_project.protocol_v2 import repetition_flags
from math_project.validation_analysis import load_config, make_jobs, paired_counts

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def evaluation_state(model, tokenizer):
    import numpy as np
    import torch
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    was_training, use_cache, padding = model.training, model.config.use_cache, tokenizer.padding_side
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            model.eval()
            model.config.use_cache = True
            yield
    finally:
        model.train(was_training)
        model.config.use_cache = use_cache
        tokenizer.padding_side = padding
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)


class GenerationMonitor:
    def __init__(self, bundle, base, output, stage, settings):
        config, self.scoring = load_config(ROOT/'configs/validation_analysis.json')
        if settings != config:
            raise ValueError('Generated-validation settings differ from the frozen Notebook 05 contract')
        self.jobs = [j for j in make_jobs(bundle, config) if j['mode'] == 'few_shot']
        if stage == 'smoke':
            self.jobs = self.jobs[:2]
        self.bundle, self.base, self.output = bundle, base, Path(output)/'generated_validation'
        self.output.mkdir(exist_ok=True)
        self.batch_size = config['batch_size']
        self.identity = {'scope': config['scope'], 'selection': [
            {'key': j['key'], 'prompt_sha256': j['prompt_sha256'],
             'reference_answer': j['row']['answer_canonical']} for j in self.jobs],
            'scoring': self.scoring, 'bundle_sha256': bundle['fingerprint']}
        path = self.output/'identity.json'
        if path.exists() and json.loads(path.read_text()) != self.identity:
            raise ValueError('Generated-validation identity changed')
        atomic_json(path, self.identity)

    def measure(self, model, label, step, epoch):
        from math_project.protocol_v2 import score_target
        records = []
        with evaluation_state(model, self.base.tokenizer):
            runner = BoundedRunner(ModelRunner(model, self.base.tokenizer, self.bundle['config'], self.base.device), self.scoring)
            for start in range(0, len(self.jobs), self.batch_size):
                batch = self.jobs[start:start+self.batch_size]
                answers = runner.generate([j['prompt'] for j in batch])
                if len(answers) != len(batch):
                    raise ValueError('Wrong generated-validation batch size')
                for job, answer in zip(batch, answers):
                    reference = job['row']['answer_canonical']
                    records.append({**answer, **score_target(answer['completion'], answer['finish_reason'], reference, self.scoring),
                        'key': job['key'], 'split': 'validation', 'mode': 'few_shot', 'category': None,
                        'question': job['row']['question'], 'reference_answer': reference,
                        'prompt_sha256': job['prompt_sha256'],
                        'repetition_flags': repetition_flags(answer['completion'])})
        metrics = group_metrics(records, len(self.jobs))
        metrics['repetition_flagged'] = sum(any(r['repetition_flags'].values()) for r in records)
        result = {'label': label, 'step': step, 'epoch': epoch, 'metrics': metrics, 'predictions': records,
                  'selection_use': 'observation_only; checkpoint selection remains validation token NLL'}
        base_path = self.output/'base.json'
        if label != 'base' and base_path.exists():
            result['paired_vs_base'] = paired_counts(json.loads(base_path.read_text())['predictions'], records)
        # Recompute an interrupted epoch observation; publish only complete batches of all questions.
        atomic_json(self.output/f'{label}.json', result)
        print('Generated validation: '+json.dumps({k:v for k,v in result.items() if k != 'predictions'}), flush=True)
        return result
