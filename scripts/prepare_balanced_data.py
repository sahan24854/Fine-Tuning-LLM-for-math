"""Prepare a checked training-data variant; never train or evaluate a model."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from math_project.balanced_data import prepare
from math_project.data import sha256
from math_project.evaluation import atomic_json, load_bundle
from scripts.train_lora import file_hashes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    from transformers import AutoTokenizer
    config=json.loads((ROOT/'configs/balanced_data.json').read_text())
    bundle=load_bundle(args.bundle)
    tokenizer=AutoTokenizer.from_pretrained(bundle['path']/'tokenizer',local_files_only=True,trust_remote_code=False)
    if tokenizer.eos_token_id is None:raise ValueError('EOS token required')
    summary=prepare(bundle,tokenizer,args.output,config)
    atomic_json(args.output/'identity.json',{'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_sha256':{n:sha256((ROOT/n).read_bytes()) for n in ('math_project/balanced_data.py','configs/balanced_data.json','scripts/prepare_balanced_data.py')}})
    atomic_json(args.output/'result_manifest.json',{'status':'balanced_data_preparation_complete',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'files_sha256':file_hashes(args.output,exclude=('result_manifest.json',))})
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
