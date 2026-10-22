"""Same-path comparisons on a real tiny CPU model; no remote weights."""
import json
from pathlib import Path
import tempfile
import unittest
from scripts.check_inference_paths import differences
ROOT=Path(__file__).resolve().parents[1]


class PathTests(unittest.TestCase):
    def test_differences_align_keys_and_check_stop_boundaries(self):
        a=[{'key':k,'completion':'x','raw_completion':'x','finish_reason':'eos','generated_tokens':1,'correct':False} for k in ('a','b')]
        self.assertEqual(differences(a,list(reversed(a))),[])
        changed=[dict(a[0]),dict(a[1])];changed[0]['finish_reason']='length'
        self.assertEqual(differences(a,changed)[0]['key'],'a')
        with self.assertRaisesRegex(ValueError,'matched'):differences(a,a[:1])

    def test_real_zero_adapter_probe_performs_no_training_and_saves_complete_results(self):
        from test_evaluation import fixture_bundle
        from test_inference import InferenceTests
        from scripts.check_inference_paths import run
        from math_project.validation_analysis import load_config
        from scripts.train_lora import verify_files
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bundle=fixture_bundle(root/'bundle')
            base=InferenceTests().make_runner(bundle)
            config,scoring=load_config(ROOT/'configs/validation_analysis.json')
            config['questions']=2;scoring['generation']['max_new_tokens']=8
            before={n:p.detach().clone() for n,p in base.model.named_parameters()}
            result=run(bundle,base,{'fixture':'cpu'},root/'paths',config,scoring)
            self.assertEqual(result['predictions'],8)
            self.assertEqual(result['optimizer_updates'],0)
            self.assertTrue(result['reproducibility_gate_passed'])
            # Original parameters remain unchanged even though wrapping renames targets.
            current=base.model.state_dict()
            targets=('q_proj','k_proj','v_proj','o_proj')
            import torch
            for name,value in before.items():
                actual=name.replace('.weight','.base_layer.weight') if any(f'.{t}.' in name for t in targets) else name
                self.assertTrue(torch.equal(current["base_model.model."+actual],value))
            verify_files(root/'paths','result_manifest.json')
            for name in ('direct_base','direct_repeat','zero_adapter','disabled_adapter'):
                data=json.loads((root/'paths'/f'{name}.json').read_text())
                self.assertEqual(len(data['predictions']),2)
                self.assertEqual({r['split'] for r in data['predictions']},{'validation'})


if __name__=='__main__':unittest.main()
