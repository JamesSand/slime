# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the math dataset to parquet format
"""

import os
import re
import datasets
import numpy as np

from verl.utils.hdfs_io import copy, makedirs
import argparse

def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split('#### ')[1].replace(',', '')
    return final_solution


def make_prefix(dp, template_type):
    problem = dp['question']
    
    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."
    
    return problem + " " + instruction_following

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/gsm8k')
    parser.add_argument('--hdfs_dir', default=None)
    # parser.add_argument('--train_size', type=int, default=7500)
    # parser.add_argument('--test_size', type=int, default=5000)
    parser.add_argument('--template_type', type=str, default='base')

    args = parser.parse_args()

    data_source = 'openai/gsm8k'

    dataset = datasets.load_dataset(data_source, 'main')

    test_dataset = dataset['test']
    
    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            random_int = np.random.randint(0, 50)
            
            question = example.pop('question')
            question = question + " " + instruction_following
            
            answer_raw = example.pop('answer')
            solution = extract_solution(answer_raw)
            if random_int == 0:
                print(f">>> Question: \n{question}")
                print(f">>> Answer: \n{answer_raw}")
                print(f">>> Solution: \n{solution}")

            data = {
                "prompt": [{
                    "role": "user",
                    "content": question
                }],
                "label": solution,
            }
            return data

        return process_fn

    test_dataset = test_dataset.map(function=make_map_fn('train'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    test_dataset.to_json(os.path.join(local_dir, 'test.jsonl'), orient='records', lines=True)

    #print length of test_dataset
    print(f"Length of test dataset: {len(test_dataset)}")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)






