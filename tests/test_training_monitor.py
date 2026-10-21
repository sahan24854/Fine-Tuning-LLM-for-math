"""Monitoring must observe training without perturbing its trajectory."""
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class MonitorTests(unittest.TestCase):
    def test_state_restored_even_when_generation_fails(self):
        import numpy as np
        import torch
        from types import SimpleNamespace
        from math_project.training_monitor import evaluation_state
        model = torch.nn.Linear(2, 2).train()
        model.config = SimpleNamespace(use_cache=False)
        tokenizer = SimpleNamespace(padding_side='right')
        random.seed(12); np.random.seed(12); torch.manual_seed(12)
        expected = (random.random(), np.random.rand(), torch.rand(1))
        random.seed(12); np.random.seed(12); torch.manual_seed(12)
        with self.assertRaisesRegex(RuntimeError, 'interrupt'):
            with evaluation_state(model, tokenizer):
                self.assertFalse(model.training)
                self.assertTrue(model.config.use_cache)
                random.random(); np.random.rand(); torch.rand(1)
                tokenizer.padding_side = 'left'
                raise RuntimeError('interrupt')
        self.assertTrue(model.training)
        self.assertFalse(model.config.use_cache)
        self.assertEqual(tokenizer.padding_side, 'right')
        actual = (random.random(), np.random.rand(), torch.rand(1))
        self.assertEqual(actual[:2], expected[:2])
        self.assertTrue(torch.equal(actual[2], expected[2]))

    def test_real_monitor_preserves_adapter_updates_across_epochs(self):
        import torch
        from safetensors.torch import load_file
        from test_training import TrainingTests
        from math_project.inference import load_runner
        from scripts.train_lora import run_training, verify_files
        from math_project.validation_analysis import load_config
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle, _, config = TrainingTests().make_training_fixture(root)
            config['epochs'] = 2
            config['smoke']['max_updates'] = 4
            config['lora']['lora_dropout'] = .1
            settings, scoring = load_config(ROOT/'configs/validation_analysis.json')
            settings['questions'] = 2
            scoring['generation']['max_new_tokens'] = 8
            plain, _ = load_runner(bundle, 'cpu')
            run_training(bundle, plain, {'fixture':'cpu'}, config, root/'plain', 'smoke')
            monitored, _ = load_runner(bundle, 'cpu')
            config['generated_validation'] = settings
            with patch('math_project.validation_analysis.load_config', return_value=(settings, scoring)), \
                 patch('math_project.training_monitor.load_config', return_value=(settings, scoring)):
                result = run_training(bundle, monitored, {'fixture':'cpu'}, config, root/'monitored', 'smoke')
            left = load_file(root/'plain/adapter/adapter_model.safetensors')
            right = load_file(root/'monitored/adapter/adapter_model.safetensors')
            self.assertTrue(all(torch.equal(left[k], right[k]) for k in left))
            self.assertEqual(set(result['generated_validation']), {'base','epoch-step-2','epoch-step-4','selected'})
            for label in result['generated_validation']:
                artifact = json.loads((root/'monitored/generated_validation'/f'{label}.json').read_text())
                self.assertEqual(len(artifact['predictions']), 2)
                self.assertTrue(all(r['split']=='validation' and r['mode']=='few_shot' for r in artifact['predictions']))
            verify_files(root/'monitored', 'result_manifest.json')

            # An epoch checkpoint already includes its observation. Recovery must
            # restore dropout RNG and keep the same eventual selected weights.
            from transformers import TrainerCallback
            from math_project.training import trainer_class
            parent = trainer_class()
            class Interrupt(TrainerCallback):
                def on_save(self, args, state, control, **kwargs):
                    if state.global_step == 2:
                        raise RuntimeError('epoch interruption')
            class InterruptedTrainer(parent):
                def __init__(self, **kwargs):
                    kwargs['callbacks'].append(Interrupt())
                    super().__init__(**kwargs)
            with patch('math_project.validation_analysis.load_config', return_value=(settings, scoring)), \
                 patch('math_project.training_monitor.load_config', return_value=(settings, scoring)):
                fresh, _ = load_runner(bundle, 'cpu')
                with patch('scripts.train_lora.trainer_class', return_value=InterruptedTrainer):
                    with self.assertRaisesRegex(RuntimeError, 'epoch interruption'):
                        run_training(bundle, fresh, {'fixture':'cpu'}, config, root/'resumed', 'smoke')
                fresh, _ = load_runner(bundle, 'cpu')
                resumed = run_training(bundle, fresh, {'fixture':'cpu'}, config, root/'resumed', 'smoke')
            recovered = load_file(root/'resumed/adapter/adapter_model.safetensors')
            self.assertTrue(all(torch.equal(right[k], recovered[k]) for k in right))
            self.assertEqual(set(resumed['generated_validation']), set(result['generated_validation']))

    def test_trial_changes_only_learning_rate_and_monitoring(self):
        original = json.loads((ROOT/'configs/lora_training.json').read_text())
        candidate = json.loads((ROOT/'configs/lora_training_lower_lr.json').read_text())
        self.assertEqual(candidate.pop('learning_rate'), 5e-5)
        candidate.pop('generated_validation')
        candidate.pop('training_id')
        original.pop('learning_rate'); original.pop('training_id')
        self.assertEqual(candidate, original)

    def test_validation_output_discovery_in_directory_and_zip(self):
        import shutil
        from scripts.find_results import find_result
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root/'output'; data.mkdir()
            (data/'result_manifest.json').write_text(json.dumps({'status':'validation_analysis_complete'}))
            self.assertEqual(find_result(root, 'validation'), data)
            archive = shutil.make_archive(str(root/'saved'), 'zip', data)
            extracted = find_result(Path(archive), 'validation')
            self.assertEqual(json.loads((extracted/'result_manifest.json').read_text())['status'], 'validation_analysis_complete')


if __name__ == '__main__':
    unittest.main()
