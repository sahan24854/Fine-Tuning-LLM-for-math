"""Validation-only scope, literal flags, resume, and real adapter CLI."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from math_project.data import sha256, write_json
from math_project.validation_analysis import equality_checks, load_config, make_jobs, paired_counts, verify_training

ROOT=Path(__file__).resolve().parents[1]
HAS_TRAINING=all(importlib.util.find_spec(n) for n in ("torch","transformers","peft","accelerate"))


class ValidationAnalysisTests(unittest.TestCase):
    def test_flags_only_whole_literal_equalities_and_never_execute_expressions(self):
        checks=equality_checks("52 + 18 = 60\n165 - 65 = 100\n0.5 * 4 = 2\n9 / 0 = 0\n3 × 4 = 12\n10 ÷ 2 = 5\n1 - -2 = 3")
        self.assertEqual([c["status"] for c in checks],["numeric_equality_mismatch","exact_equality","exact_equality","undefined_division","exact_equality","exact_equality","exact_equality"])
        for text in ("1+2+3=6", "Explanation: 18 + 9 = 27", "18 + 9 = 27 dollars", "__import__('os').system('false')", "x + 1 = 2"):
            self.assertEqual(equality_checks(text),[])

    def test_selection_is_bounded_validation_only_and_independent_of_order_or_correctness(self):
        config,_=load_config(ROOT/"configs/validation_analysis.json")
        rows=[{"id":f"validation/{i}","question":f"Question {i}","answer_canonical":str(i)} for i in range(70)]
        bundle={"splits":{"validation":rows,"gsm8k_test":[{"id":"must_not_select"}]},"demonstrations":[]}
        jobs=make_jobs(bundle,config)
        self.assertEqual(len(jobs),100)
        self.assertEqual({j["split"] for j in jobs},{"validation"})
        self.assertEqual(len({j["row"]["id"] for j in jobs}),50)
        before=[j["key"] for j in jobs]
        rows.reverse()
        for row in rows:row["answer_canonical"]="changed"
        self.assertEqual([j["key"] for j in make_jobs(bundle,config)],before)
        with self.assertRaisesRegex(ValueError,"at most 50"):
            make_jobs(bundle,{**config,"questions":51,"max_questions":100})
        with self.assertRaisesRegex(ValueError,"validation-only"):
            make_jobs(bundle,{**config,"scope":"test"})

    def test_paired_counts_align_keys_instead_of_position(self):
        base=[{"key":"a","correct":True},{"key":"b","correct":False}]
        tuned=[{"key":"b","correct":True},{"key":"a","correct":False}]
        self.assertEqual(paired_counts(base,tuned),{"paired":2,"improved":1,"regressed":1,"both_correct":0,"both_incorrect":0})


@unittest.skipUnless(HAS_TRAINING,"requires training environment")
class ValidationPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_training import TrainingTests
        from math_project.evaluation import load_bundle
        from math_project.inference import load_runner
        from scripts.train_lora import run_training
        cls.temp=tempfile.TemporaryDirectory()
        cls.root=Path(cls.temp.name)
        bundle,base,training_config=TrainingTests().make_training_fixture(cls.root)
        base.model.config.max_position_embeddings=2048
        base.model.generation_config.eos_token_id=list(range(base.model.config.vocab_size))
        base.model.save_pretrained(cls.root/"tiny-model")
        bundle["config"]["context_window_tokens"]=2048
        write_json(bundle["path"]/"config.resolved.json",bundle["config"])
        bundle["manifest"]["files_sha256"]={str(p.relative_to(bundle["path"])):sha256(p.read_bytes()) for p in bundle["path"].rglob("*") if p.is_file() and p.name!="manifest.json"}
        write_json(bundle["path"]/"manifest.json",bundle["manifest"])
        cls.bundle=load_bundle(bundle["path"])
        base,runtime=load_runner(cls.bundle,"cpu")
        run_training(cls.bundle,base,runtime,training_config,cls.root/"smoke","smoke")
        base,runtime=load_runner(cls.bundle,"cpu")
        run_training(cls.bundle,base,runtime,training_config,cls.root/"training","full",cls.root/"smoke")
        cls.training=verify_training(cls.bundle,cls.root/"training")
        config,cls.scoring=load_config(ROOT/"configs/validation_analysis.json")
        cls.config={**config,"questions":2}
        cls.runtime={"fixture":"cpu"}

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_interruption_resume_report_identity_and_review_template(self):
        from scripts.analyze_validation import finalize,run_stage
        from test_pilot import Runner
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/"analysis"
            # One batch contains all four jobs; a failed generation writes none.
            with self.assertRaisesRegex(RuntimeError,"interrupted"):
                run_stage(self.bundle,self.training,self.config,self.scoring,output,"base",Runner(fail_after=0),self.runtime)
            self.assertFalse((output/"base/result_manifest.json").exists())
            runner=Runner()
            run_stage(self.bundle,self.training,self.config,self.scoring,output,"base",runner,self.runtime)
            self.assertFalse(json.loads((output/"summary.json").read_text())["execution_complete"])
            run_stage(self.bundle,self.training,self.config,self.scoring,output,"tuned",runner,self.runtime)
            summary=finalize(self.bundle,self.training,self.config,self.scoring,output)
            self.assertTrue(summary["execution_complete"])
            self.assertEqual(summary["predictions"],8)
            self.assertEqual(summary["base_vs_tuned_by_mode"]["zero_shot"]["paired"],2)
            before=(output/"result_manifest.json").read_bytes()
            calls=runner.calls
            run_stage(self.bundle,self.training,self.config,self.scoring,output,"base",runner,self.runtime)
            self.assertEqual(runner.calls,calls)
            self.assertEqual(finalize(self.bundle,self.training,self.config,self.scoring,output),summary)
            self.assertEqual((output/"result_manifest.json").read_bytes(),before)
            self.assertEqual(len((output/"paired_predictions.jsonl").read_bytes().splitlines()),4)
            self.assertIn("review_category",(output/"review_template.csv").read_text())
            with self.assertRaisesRegex(ValueError,"identity changed"):
                run_stage(self.bundle,self.training,{**self.config,"seed":43},self.scoring,output,"base",runner,self.runtime)
            (output/"report.html").write_text('changed')
            with self.assertRaisesRegex(ValueError,"checksum"):
                finalize(self.bundle,self.training,self.config,self.scoring,output)

    def test_real_cli_both_modes_both_models_and_no_test_questions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            write_json(root/"config.json",self.config)
            command=[sys.executable,str(ROOT/"scripts/analyze_validation.py"),"--bundle",str(self.bundle["path"]),
                "--training",str(self.root/"training"),"--output",str(root/"analysis"),"--config",str(root/"config.json"),"--device","cpu"]
            for stage in ("verify","base","tuned","report"):
                result=subprocess.run(command+["--stage",stage],capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            summary=json.loads((root/"analysis/summary.json").read_text())
            self.assertEqual(summary["predictions"],8)
            for role in ("base","tuned"):
                rows=[json.loads(line) for line in (root/f"analysis/{role}/predictions.jsonl").read_bytes().splitlines()]
                self.assertEqual(len(rows),4)
                self.assertEqual({r["split"] for r in rows},{"validation"})
                self.assertEqual({r["mode"] for r in rows},{"zero_shot","few_shot"})
                self.assertTrue(all(r["generated_tokens"]==1 for r in rows))
