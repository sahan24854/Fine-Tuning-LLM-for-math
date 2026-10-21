"""Verify historical comparison inputs, paired joins, and refusal of changed scores."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from math_project.data import sha256, write_json, write_jsonl
from math_project.pilot import group_metrics
from math_project.protocol_v2 import score_target, repetition_flags
from math_project.validation_analysis import load_config, make_jobs
from scripts.train_lora import file_hashes
from scripts.compare_training_trial import compare, load_control, CONTROL_TRAINING_SHA
from math_project.comparison import FROZEN_EVALUATOR_FILES

ROOT = Path(__file__).resolve().parents[1]


class TrialTests(unittest.TestCase):
    def fixture(self, root):
        config, scoring = load_config(ROOT/'configs/validation_analysis.json')
        bundle = {'fingerprint':'fixture', 'config':{'model':{'id':'tiny','revision':'fixed'}},
            'demonstrations':[], 'splits':{'validation':[
                {'id':f'validation/{i}', 'question':f'{i} plus 1?', 'answer_canonical':str(i+1)} for i in range(50)]}}
        jobs = make_jobs(bundle, config)
        records = []
        for j in jobs:
            text = f"Final answer: {j['row']['answer_canonical']}\n"
            records.append({'completion':text, 'raw_completion':text, 'finish_reason':'final_answer_line',
                'generated_tokens':5, 'prompt_tokens':20, 'amortized_generation_seconds':.01,
                **score_target(text,'final_answer_line',j['row']['answer_canonical'],scoring),
                **{k:j[k] for k in ('key','split','mode','category','prompt_sha256')},
                'question':j['row']['question'], 'reference_answer':j['row']['answer_canonical'],
                'repetition_flags':repetition_flags(text)})
        previous=root/'previous';previous.mkdir()
        identity={'bundle_sha256':'fixture','base_model':bundle['config']['model'], 'config':config,'scoring':scoring,
            'training_artifact_sha256':CONTROL_TRAINING_SHA,
            'code_files_sha256':{n:sha256((ROOT/n).read_bytes()) for n in FROZEN_EVALUATOR_FILES}}
        write_json(previous/'analysis_identity.json',identity)
        for role in ('base','tuned'):
            (previous/role).mkdir()
            write_jsonl(previous/role/'predictions.jsonl',records)
            write_json(previous/role/'identity.json',{'role':role,'runtime':{'fixture':'cpu'},
                'analysis_identity_sha256':sha256(json.dumps(identity,sort_keys=True).encode())})
        write_json(previous/'result_manifest.json',{'status':'validation_analysis_complete',
            'files_sha256':file_hashes(previous)})
        training=root/'training';training.mkdir();(training/'generated_validation').mkdir()
        write_json(training/'identity.json', {'training_config':json.loads((ROOT/'configs/lora_training_lower_lr.json').read_text()),
            'code_files_sha256':{n:sha256((ROOT/n).read_bytes()) for n in FROZEN_EVALUATOR_FILES},'runtime':{'fixture':'cpu'}})
        selected=[j for j in jobs if j['mode']=='few_shot']
        rows=[r for r in records if r['mode']=='few_shot']
        write_json(training/'generated_validation/identity.json', {'scope':config['scope'], 'scoring':scoring,
            'bundle_sha256':'fixture','selection':[{'key':j['key'],'prompt_sha256':j['prompt_sha256'],
            'reference_answer':j['row']['answer_canonical']} for j in selected]})
        for label in ('base','selected'):
            write_json(training/'generated_validation'/f'{label}.json', {'label':label,'step':0 if label=='base' else 633,
                'predictions':list(reversed(rows)), 'metrics':{**group_metrics(rows,50),'repetition_flagged':0}})
        return bundle, previous, training

    def test_comparison_joins_by_key_and_refuses_changed_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); bundle,previous,training=self.fixture(root)
            # Full adapter checksum/loading contracts are covered by training/validation tests.
            with patch('scripts.compare_training_trial.verify_training',return_value={
                    'fingerprint':'candidate','summary':{'best_checkpoint':'checkpoint-633'}}):
                summary=compare(bundle,training,previous,root/'report')
                self.assertEqual(summary['paired_vs_first_adapter']['both_correct'],50)
                self.assertEqual(summary['base_completion_differences'],0)
                path=training/'generated_validation/selected.json'
                data=json.loads(path.read_text());data['predictions'][0]['correct']=False;write_json(path,data)
                with self.assertRaisesRegex(ValueError,'score or metadata'):
                    compare(bundle,training,previous,root/'bad')

    def test_control_checksums_and_first_adapter_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bundle,previous,_=self.fixture(root)
            load_control(bundle,previous)
            path=previous/'analysis_identity.json'
            identity=json.loads(path.read_text());identity['training_artifact_sha256']='another adapter';write_json(path,identity)
            with self.assertRaisesRegex(ValueError,'checksum'):
                load_control(bundle,previous)
            write_json(previous/'result_manifest.json',{'status':'validation_analysis_complete',
                'files_sha256':file_hashes(previous,exclude=('result_manifest.json',))})
            with self.assertRaisesRegex(ValueError,'reviewed Notebook 05'):
                load_control(bundle,previous)


if __name__ == '__main__':unittest.main()
