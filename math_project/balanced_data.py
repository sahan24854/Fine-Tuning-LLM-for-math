"""Deterministic elementary examples with independently checked integer arithmetic."""
from collections import Counter
import json
from pathlib import Path
import random
import shutil
import subprocess

from math_project.data import completion, prompt, question_key, read_jsonl, sha256, write_jsonl
from math_project.evaluation import atomic_json, load_bundle

ROOT = Path(__file__).resolve().parents[1]
SKILLS = ('add', 'subtract', 'equal_groups', 'divide', 'two_step_total', 'two_step_remaining')


def render(skill, a, b, c, template):
    """Use closed, reviewed grammars. Never evaluate expressions or generated prose."""
    if not (1 <= a <= 99 and 1 <= b <= 20 and 1 <= c <= 20 and template in (0, 1, 2)):
        raise ValueError('Parameters outside declared integer ranges')
    if skill == 'add':
        questions = [f'Mina has {a} marbles and receives {b} more marbles. How many marbles does Mina have now?',
            f'A shelf holds {a} books. A librarian adds {b} books. How many books are on the shelf now?',
            f'A bus has {a} passengers. Another {b} passengers board and nobody leaves. How many passengers are on the bus?']
        steps = [(a, '+', b, a+b)]
    elif skill == 'subtract':
        if a < b: raise ValueError('Subtraction would be negative')
        questions = [f'Mina has {a} marbles and gives away {b} marbles. How many marbles does Mina have left?',
            f'A shelf holds {a} books. A librarian removes {b} books. How many books remain on the shelf?',
            f'A bus has {a} passengers. Then {b} passengers leave and nobody boards. How many passengers remain on the bus?']
        steps = [(a, '-', b, a-b)]
    elif skill == 'equal_groups':
        questions = [f'There are {a} boxes with {b} pencils in each box. How many pencils are there altogether?',
            f'A shop has {a} bags containing {b} apples per bag. How many apples are in the bags?',
            f'A gardener plants {a} rows with {b} seeds in each row. How many seeds does the gardener plant?']
        steps = [(a, '*', b, a*b)]
    elif skill == 'divide':
        total = a*b
        questions = [f'{total} pencils are shared equally among {b} children. How many pencils does each child receive?',
            f'A shop packs {total} apples equally into {b} bags. How many apples go into each bag?',
            f'A gardener distributes {total} seeds equally across {b} rows. How many seeds go into each row?']
        steps = [(total, '/', b, a)]
    elif skill == 'two_step_total':
        questions = [f'There are {a} boxes with {b} pencils in each box and {c} loose pencils. How many pencils are there altogether?',
            f'A shop has {a} bags with {b} apples in each bag and another {c} apples on a tray. How many apples does it have in total?',
            f'A gardener plants {a} rows of {b} seeds and then plants {c} extra seeds. How many seeds are planted in total?']
        steps = [(a, '*', b, a*b), (a*b, '+', c, a*b+c)]
    elif skill == 'two_step_remaining':
        if a*b < c: raise ValueError('Cannot remove more objects than exist')
        questions = [f'There are {a} boxes with {b} pencils in each box. Mina gives away {c} pencils from this supply. How many pencils remain?',
            f'A shop has {a} bags with {b} apples per bag. It sells {c} apples in total. How many apples remain?',
            f'A gardener has {a} packets of {b} seeds each and plants {c} seeds in total. How many seeds remain unused?']
        steps = [(a, '*', b, a*b), (a*b, '-', c, a*b-c)]
    else:
        raise ValueError('Unknown elementary skill')
    return questions[template], steps


def make_row(skill, params, template, split, index):
    question, steps = render(skill, *params, template)
    answer = str(steps[-1][-1])
    return {'id':f'elementary-v1/{split}/{skill}/{index:05d}', 'dataset':'elementary_synthetic_v1',
        'skill':skill, 'template_id':f'{skill}/{template}', 'parameters':list(params),
        'question':question, 'solution':'\n'.join(f'{a} {op} {b} = {v}' for a,op,b,v in steps),
        'answer':answer, 'answer_canonical':answer, 'checked_steps':[list(s) for s in steps]}


def verify_row(row):
    skill = row['skill']
    template = int(row['template_id'].split('/')[-1])
    q, steps = render(skill, *row['parameters'], template)
    if (row['question'] != q or row['checked_steps'] != [list(s) for s in steps] or
        row['solution'] != '\n'.join(f'{a} {op} {b} = {v}' for a,op,b,v in steps) or
        row['answer'] != str(steps[-1][-1]) or row['answer_canonical'] != row['answer']):
        raise ValueError('Synthetic text, calculation or label changed')
    for a, op, b, value in row['checked_steps']:
        valid = (a+b == value if op == '+' else a-b == value if op == '-' else
                 a*b == value if op == '*' else b != 0 and a == b*value)
        if not valid: raise ValueError('Invalid integer arithmetic')


def generate(config, forbidden=()):
    if tuple(config['skills']) != SKILLS or config['synthetic_train_rows'] % (2*len(SKILLS)):
        raise ValueError('Require balanced skills and two training templates per skill')
    count = config['synthetic_train_rows']//len(SKILLS)
    if not 2 <= count <= 1000 or not 1 <= config['diagnostic_rows_per_skill'] <= 100:
        raise ValueError('Unsupported generation budget')
    rng = random.Random(config['seed'])
    questions = set(forbidden)
    # Split before wording: identical numerical tuples for a skill never cross splits.
    tuples = set()
    train, diagnostic = [], []
    for split, target, n in [('train',train,count), ('diagnostic',diagnostic,config['diagnostic_rows_per_skill'])]:
        for skill in SKILLS:
            for i in range(n):
                for attempt in range(100000):
                    params = (rng.randint(1,99),rng.randint(1,20),rng.randint(1,20))
                    # Ignore unused c when detecting mathematical duplicates.
                    key = (skill,*params[:(3 if skill.startswith('two_step') else 2)])
                    if key in tuples: continue
                    template = i%2 if split == 'train' else 2
                    try:row = make_row(skill, params, template, split, i)
                    except ValueError:continue
                    normalized = question_key(row['question'])
                    if normalized in questions: continue
                    verify_row(row)
                    tuples.add(key);questions.add(normalized);target.append(row)
                    break
                else:raise ValueError('Could not construct unique checked data')
    return train, diagnostic


def encode(row, tokenizer, max_tokens):
    row = dict(row)
    row.update(prompt=prompt(row['question']), completion=completion(row))
    row['prompt_token_count'] = len(tokenizer.encode(row['prompt'],add_special_tokens=False))
    row['completion_token_count'] = len(tokenizer.encode(row['completion'],add_special_tokens=False))+1
    if row['prompt_token_count']+row['completion_token_count'] > max_tokens:
        raise ValueError('Synthetic example exceeds budget; no truncation allowed')
    return row


def prepare(bundle, tokenizer, output, config):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Preparation output must be empty; preserve completed artifacts')
    output.mkdir(parents=True,exist_ok=True)
    original = read_jsonl(bundle['path']/'train.jsonl')
    # Check against every existing training/development/test question, without using correctness.
    forbidden = {question_key(r['question']) for rows in [original,*bundle['splits'].values()] for r in rows}
    synthetic, diagnostic = generate(config, forbidden)
    keep = len(original)-len(synthetic)
    mandatory = {r['id'] for r in bundle['demonstrations']}
    if keep <= len(mandatory):raise ValueError('Original training set too small for this mixture')
    rank = lambda r:(sha256(f"{config['seed']}:{r['id']}".encode()),r['id'])
    retained = [r for r in original if r['id'] in mandatory]
    retained += sorted([r for r in original if r['id'] not in mandatory],key=rank)[:keep-len(retained)]
    encoded = [encode(r,tokenizer,bundle['config']['max_training_tokens']) for r in synthetic]
    mixed = sorted(retained+encoded,key=rank)
    variant = output/'bundle'
    shutil.copytree(bundle['path'],variant)
    write_jsonl(variant/'train.jsonl',mixed)
    write_jsonl(output/'synthetic_train.jsonl',synthetic)
    write_jsonl(output/'synthetic_diagnostic.jsonl',diagnostic)
    write_jsonl(output/'review_sample.jsonl',[next(r for r in synthetic if r['template_id']==f'{skill}/{template}')
        for skill in SKILLS for template in (0,1)])
    manifest = dict(bundle['manifest'])
    manifest.update(parent_bundle_sha256=bundle['fingerprint'], training_variant=config)
    manifest['files_sha256'] = {str(p.relative_to(variant)):sha256(p.read_bytes()) for p in sorted(variant.rglob('*'))
                               if p.is_file() and p.name!='manifest.json'}
    atomic_json(variant/'manifest.json',manifest)
    new = load_bundle(variant)
    original_tokens = sum(r['completion_token_count'] for r in original)
    mixed_tokens = sum(r['completion_token_count'] for r in mixed)
    fixed = ['validation.jsonl','gsm8k_test.jsonl','svamp_test.jsonl','arithmetic_test.jsonl','demonstrations.json','config.resolved.json']
    if any(sha256((variant/n).read_bytes()) != bundle['manifest']['files_sha256'][n] for n in fixed):
        raise ValueError('Evaluation inputs changed')
    summary = {'status':'balanced_data_prepared_not_trained','parent_bundle_sha256':bundle['fingerprint'],
        'variant_bundle_sha256':new['fingerprint'], 'original_rows':len(original),'mixed_rows':len(mixed),
        'retained_gsm8k':len(retained),'synthetic_rows':len(synthetic),
        'synthetic_per_skill':dict(Counter(r['skill'] for r in synthetic)),
        'diagnostic_per_skill':dict(Counter(r['skill'] for r in diagnostic)),
        'original_supervised_tokens_per_epoch':original_tokens,'mixed_supervised_tokens_per_epoch':mixed_tokens,
        'same_row_count':len(original)==len(mixed), 'same_supervised_token_budget':original_tokens==mixed_tokens,
        'unchanged_evaluation_files':fixed,'config':config,
        'interpretation':'Training-data mixture intervention, including simpler wording and shorter worked solutions. Equal optimizer updates do not imply equal supervised-token exposure. Synthetic diagnostics are limited template-family development checks, not broad maths evidence.'}
    atomic_json(output/'summary.json',summary)
    return summary
