import evaluate as hf_evaluate
import os
import sys
from sanitize import sanitize

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
pass_at_k = hf_evaluate.load("code_eval")

def pass_at_1(references, predictions):
    return pass_at_k.compute(
        references=references,
        predictions=predictions,
        k=[1],
    )[0]["pass@1"]

import json

        
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

file_path = sys.argv[1]
data = read_jsonl(file_path)

references = [sample['target'] for sample in data]

predictions = [[sanitize(sample['doc']['prompt'] + "\n" + sample['resps'][0][0].split('```python\n', 1)[-1].split('```')[0], 
                sample['doc']["entry_point"])] 
                for sample in data]

pass_at_1s = [pass_at_1([reference], [prediction]) for reference, prediction in zip(references, predictions)]
print(sum(pass_at_1s)/len(pass_at_1s))

def write_jsonl(data, file_path):
    with open(file_path, 'w') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')

res = [{"task_id": sample['doc']['task_id'], "completion": pred, "pass_at_1": res} 
       for sample, pred, res  in zip(data, predictions, pass_at_1s)]
write_jsonl(res, file_path+'.cleaned')