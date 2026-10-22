"""Arithmetic correctness, leakage barriers and immutable original evaluation inputs."""
from collections import Counter
import copy
import json
from pathlib import Path
import tempfile
import unittest
from math_project.balanced_data import SKILLS, encode, generate, prepare, verify_row
from math_project.data import question_key, read_jsonl, sha256, write_json, write_jsonl
from math_project.evaluation import load_bundle
ROOT=Path(__file__).resolve().parents[1]


class Tokenizer:
    eos_token_id=2
    def encode(self,text,add_special_tokens=False):return list(range(len(text.split())))


class BalancedTests(unittest.TestCase):
    def config(self):return json.loads((ROOT/'configs/balanced_data.json').read_text())

    def test_complete_dataset_is_balanced_correct_and_held_out_by_tuple_and_wording(self):
        config=self.config()
        train,dev=generate(config)
        self.assertEqual(len(train),1680);self.assertEqual(len(dev),120)
        self.assertEqual(Counter(r['skill'] for r in train),dict.fromkeys(SKILLS,280))
        self.assertEqual(Counter(r['skill'] for r in dev),dict.fromkeys(SKILLS,20))
        self.assertEqual((train,dev),generate(config))
        self.assertEqual(len({question_key(r['question']) for r in train+dev}),1800)
        key=lambda r:(r['skill'],*r['parameters'][:3 if r['skill'].startswith('two_step') else 2])
        self.assertFalse({key(r) for r in train}&{key(r) for r in dev})
        self.assertFalse({r['template_id'] for r in train}&{r['template_id'] for r in dev})
        for row in train+dev:
            verify_row(row)
            a,b,c=row['parameters'];skill=row['skill']
            expected={'add':a+b,'subtract':a-b,'equal_groups':a*b,'divide':a,
                'two_step_total':a*b+c,'two_step_remaining':a*b-c}[skill]
            self.assertEqual(int(row['answer']),expected)
            self.assertGreaterEqual(expected,0)
        changed=copy.deepcopy(train[0]);changed['answer']='-999'
        with self.assertRaisesRegex(ValueError,'changed'):verify_row(changed)
        train2,_=generate(config,{question_key(train[0]['question'])})
        self.assertNotIn(train[0]['question'],{r['question'] for r in train2})

    def test_variant_keeps_count_demonstrations_and_every_evaluation_byte(self):
        from test_evaluation import fixture_bundle
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bundle=fixture_bundle(root/'original')
            config=bundle['config'];config['max_training_tokens']=1024
            write_json(bundle['path']/'config.resolved.json',config)
            train=[encode({'id':f'train/{i}','question':f'A toy question numbered {i}?',
                'solution':f'{i} + 1 = {i+1}','answer':str(i+1),'answer_canonical':str(i+1)},Tokenizer(),1024) for i in range(48)]
            write_jsonl(bundle['path']/'train.jsonl',train)
            write_json(bundle['path']/'demonstrations.json',[train[0]])
            manifest=bundle['manifest'];manifest['counts']['train']=48
            manifest['files_sha256']={str(p.relative_to(bundle['path'])):sha256(p.read_bytes())
                for p in bundle['path'].rglob('*') if p.is_file() and p.name!='manifest.json'}
            write_json(bundle['path']/'manifest.json',manifest)
            bundle=load_bundle(bundle['path'])
            settings=self.config();settings['synthetic_train_rows']=12;settings['diagnostic_rows_per_skill']=2
            summary=prepare(bundle,Tokenizer(),root/'prepared',settings)
            variant=load_bundle(root/'prepared/bundle')
            self.assertEqual(summary['mixed_rows'],48)
            self.assertEqual(summary['retained_gsm8k'],36)
            self.assertNotEqual(variant['fingerprint'],bundle['fingerprint'])
            self.assertEqual(variant['demonstrations'],bundle['demonstrations'])
            for name in summary['unchanged_evaluation_files']:
                self.assertEqual((variant['path']/name).read_bytes(),(bundle['path']/name).read_bytes())
            self.assertEqual(len(read_jsonl(variant['path']/'train.jsonl')),48)
            with self.assertRaisesRegex(ValueError,'empty'):
                prepare(bundle,Tokenizer(),root/'prepared',settings)

    def test_no_silent_sequence_truncation(self):
        train,_=generate(self.config())
        with self.assertRaisesRegex(ValueError,'truncation'):
            encode(train[0],Tokenizer(),2)


if __name__=='__main__':unittest.main()
