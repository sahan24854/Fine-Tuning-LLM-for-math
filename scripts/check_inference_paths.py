"""Small same-process inference-path diagnostic, with zero optimizer updates."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from math_project.data import sha256
from math_project.evaluation import atomic_json, load_bundle
from math_project.inference import load_runner
from math_project.inference_v2 import BoundedRunner
from math_project.pilot import group_metrics
from math_project.protocol_v2 import repetition_flags, score_target
from math_project.training_monitor import evaluation_state
from math_project.validation_analysis import load_config, make_jobs
from scripts.run_pilot import validate_records
from scripts.train_lora import file_hashes


def differences(left,right):
    lookup={r['key']:r for r in left}
    if len(lookup)!=len(left) or set(lookup)!={r['key'] for r in right} or len(left)!=len(right):
        raise ValueError('Inference comparisons require exactly matched unique keys')
    return [{'key':r['key'],'left_correct':lookup[r['key']]['correct'],'right_correct':r['correct'],
             'left_finish':lookup[r['key']]['finish_reason'],'right_finish':r['finish_reason']}
            for r in right if any(r[k]!=lookup[r['key']][k] for k in ('completion','raw_completion','finish_reason','generated_tokens'))]


def observe(base,bundle,jobs,scoring):
    records=[]
    with evaluation_state(base.model,base.tokenizer):
        runner=BoundedRunner(base,scoring)
        for offset in range(0,len(jobs),4):
            batch=jobs[offset:offset+4]
            answers=runner.generate([j['prompt'] for j in batch])
            if len(answers)!=len(batch):raise ValueError('Wrong diagnostic batch size')
            for j,a in zip(batch,answers):
                reference=j['row']['answer_canonical']
                records.append({**a,**score_target(a['completion'],a['finish_reason'],reference,scoring),
                    **{k:j[k] for k in ('key','split','mode','category','prompt_sha256')},
                    'question':j['row']['question'],'reference_answer':reference,
                    'repetition_flags':repetition_flags(a['completion'])})
    validate_records(records,jobs,scoring)
    return records


def run(bundle,base,runtime,output,config,scoring):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model, __version__ as peft_version
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a fresh diagnostic output directory; retain completed results')
    output.mkdir(parents=True,exist_ok=True)
    jobs=[j for j in make_jobs(bundle,config) if j['mode']=='few_shot']
    rows={}
    generation_before=base.model.generation_config.to_dict()
    def measure(name,context=None):
        with context if context is not None else nullcontext():
            records=observe(base,bundle,jobs,scoring)
        atomic_json(output/f'{name}.json',{'predictions':records,'metrics':group_metrics(records,len(jobs))})
        rows[name]=records
        print(f'{name}: {sum(r["correct"] for r in records)}/{len(jobs)} correct',flush=True)
    # Identical prompts, ordering, padding/batching, generation, hardware and model.
    measure('direct_base')
    measure('direct_repeat')
    training=json.loads((ROOT/'configs/lora_training.json').read_text())
    torch.manual_seed(training['seed'])
    base.model=get_peft_model(base.model,LoraConfig(task_type=TaskType.CAUSAL_LM,
        revision=bundle['config']['model']['revision'],**training['lora']))
    # PEFT's default zero B gives an exactly zero adapter update at initialization.
    if not all(bool((p==0).all()) for n,p in base.model.named_parameters() if '.lora_B.' in n):
        raise ValueError('Diagnostic requires a zero-initialized adapter')
    measure('zero_adapter')
    measure('disabled_adapter',base.model.disable_adapter())
    if generation_before!=base.model.generation_config.to_dict():
        raise ValueError('Wrapping changed generation configuration')
    comparisons={name:differences(rows['direct_base'],rows[name]) for name in ('direct_repeat','zero_adapter','disabled_adapter')}
    summary={'status':'inference_path_diagnostic_complete','scope':'validation_only_no_training',
        'questions':len(jobs),'predictions':4*len(jobs), 'optimizer_updates':0,
        'groups':{name:group_metrics(records,len(jobs)) for name,records in rows.items()},
        'differences_vs_direct':comparisons,'reproducibility_gate_passed':not any(comparisons.values()),
        'interpretation':'A mismatch identifies a path-dependent output, not its numerical root cause. Matching this probe does not independently reproduce all Trainer preparation or historical runs. Do not start the next training comparison until the diagnostic is reviewed.'}
    atomic_json(output/'summary.json',summary)
    atomic_json(output/'identity.json',{'bundle_sha256':bundle['fingerprint'],'model':bundle['config']['model'],
        'runtime':{**runtime,'peft':peft_version,'model_eos_ids':base.eos_ids,'pad_token_id':base.tokenizer.pad_token_id,
            'tokenizer_eos_id':base.tokenizer.eos_token_id,'stopping':'protocol_v2_per_sequence'},
        'scoring':scoring,'selection':[{'key':j['key'],'prompt_sha256':j['prompt_sha256']} for j in jobs],
        'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_sha256':{n:sha256((ROOT/n).read_bytes()) for n in ('scripts/check_inference_paths.py',
            'math_project/inference.py','math_project/inference_v2.py','math_project/training_monitor.py',
            'math_project/protocol_v2.py','configs/lora_training.json','configs/validation_analysis.json')}})
    atomic_json(output/'result_manifest.json',{'status':'inference_path_diagnostic_complete',
        'files_sha256':file_hashes(output,exclude=('result_manifest.json',))})
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args=p.parse_args()
    if args.device=='cuda':os.environ['CUDA_VISIBLE_DEVICES']=os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')[0]
    bundle=load_bundle(args.bundle)
    config,scoring=load_config(ROOT/'configs/validation_analysis.json')
    base,runtime=load_runner(bundle,args.device)
    print(json.dumps(run(bundle,base,runtime,args.output,config,scoring),indent=2))


if __name__=='__main__':main()
