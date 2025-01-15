import argparse
import numpy as np
from tqdm import tqdm
from pebble import ProcessPool
from concurrent.futures import TimeoutError

from grader import *

from parser import *
from utils.utils import load_jsonl
from python_executor import PythonExecutor


def evaluate(data_name, prompt_type, samples: list=None, file_path: str=None, max_num_samples=None, execute=False):
    assert samples or file_path, "samples or file_path must be provided"
    if not samples:
        samples = list(load_jsonl(file_path))
    if 'idx' in samples[0]:
        samples = {sample['idx']: sample for sample in samples}.values()
        samples = sorted(samples, key=lambda x: x['idx']) 
    else:
        samples = [dict(idx=idx, **sample) for idx, sample in enumerate(samples)]

    if max_num_samples:
        print(f"max_num_samples: {max_num_samples} / {len(samples)}")
        samples = samples[:max_num_samples]
    
    # parse gt
    for sample in samples:
        sample['gt_cot'], sample['gt'] = parse_ground_truth(sample, data_name)
    params = [(idx, pred, sample['gt']) for idx, sample in enumerate(samples) for pred in sample['pred']]

    scores = []
    timeout_cnt = 0 

    with ProcessPool(max_workers=1) as pool:
        future = pool.map(math_equal_process, params, timeout=3)
        iterator = future.result()
        with tqdm(total=len(samples), desc="Evaluate") as progress_bar:
            while True:
                try:
                    result = next(iterator)
                    scores.append(result)
                except StopIteration:
                    break
                except TimeoutError as error:
                    print(error)
                    scores.append(False)
                    timeout_cnt += 1
                except Exception as error:
                    print(error.traceback)
                    exit()
                progress_bar.update(1) 

    idx = 0
    score_mat = []
    for sample in samples:
        sample['score'] = scores[idx: idx+len(sample['pred'])]
        assert len(sample['score']) == len(sample['pred'])
        score_mat.append(sample['score'])
        idx += len(sample['pred'])

    max_len = max([len(s) for s in score_mat])

    for i, s in enumerate(score_mat):
        if len(s) < max_len:
            score_mat[i] = s + [s[-1]] * (max_len - len(s)) # pad

    # output mean of each column of scores
    col_means= np.array(score_mat).mean(axis=0)
    mean_score = list(np.round(col_means * 100, decimals=1))

    result_json = {
        "num_samples": len(samples),
        "num_scores": len(scores),
        "timeout_samples": timeout_cnt,
        "empty_samples": len([s for s in samples if not s['pred'][-1]]),
        "acc": mean_score[0]
    }

    # each type score
    if "type" in samples[0]:
        type_scores = {}
        for sample in samples:
            if sample['type'] not in type_scores:
                type_scores[sample['type']] = []
            type_scores[sample['type']].append(sample['score'][-1])
        type_scores = {k: np.round(np.array(v).mean() * 100, decimals=1) for k, v in type_scores.items()}
        type_scores = {k: v for k, v in sorted(type_scores.items(), key=lambda item: item[0])}
        result_json['type_acc'] = type_scores

    # each level score
    if "level" in samples[0]:
        level_scores = {}
        for sample in samples:
            if sample['level'] not in level_scores:
                level_scores[sample['level']] = []
            level_scores[sample['level']].append(sample['score'][-1])
        level_scores = {k: np.round(np.array(v).mean() * 100, decimals=1) for k, v in level_scores.items()}
        level_scores = {k: v for k, v in sorted(level_scores.items(), key=lambda item: item[0])}
        result_json['level_acc'] = level_scores
        
    # each subject score
    if "subject" in samples[0]:
        subject_scores = {}
        for sample in samples:
            if sample['subject'] not in subject_scores:
                subject_scores[sample['subject']] = []
            subject_scores[sample['subject']].append(sample['score'][-1])
        subject_scores = {k: np.round(np.array(v).mean() * 100, decimals=1) for k, v in subject_scores.items()}
        subject_scores = {k: v for k, v in sorted(subject_scores.items(), key=lambda item: item[0])}
        result_json['subject_acc'] = subject_scores
    
    # each subfield score
    if "subfield" in samples[0]:
        subfield_scores = {}
        for sample in samples:
            if sample['subfield'] not in subfield_scores:
                subfield_scores[sample['subfield']] = []
            subfield_scores[sample['subfield']].append(sample['score'][-1])
        subfield_scores = {k: np.round(np.array(v).mean() * 100, decimals=1) for k, v in subfield_scores.items()}
        subfield_scores = {k: v for k, v in sorted(subfield_scores.items(), key=lambda item: item[0])}
        result_json['subfield_acc'] = subfield_scores
    
    print(result_json)
    return samples, result_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="math")
    parser.add_argument("--prompt_type", type=str, default="tool-integrated")
    parser.add_argument("--file_path", type=str, default=None, required=True)
    parser.add_argument("--max_num_samples", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    evaluate(data_name=args.data_name, prompt_type=args.prompt_type, file_path=args.file_path,
             max_num_samples=args.max_num_samples, execute=args.execute)
