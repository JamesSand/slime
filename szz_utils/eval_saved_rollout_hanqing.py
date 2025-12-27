
#!/usr/bin/env python3
"""
评估单个保存的 rollout .pt 文件的 math score
"""

# reference from here
# https://github.com/zhuhanqing/Lucky_RL/blob/ed3e2fdd7e9eafa4d1e6796ecc72286ffb9208f8/verl/utils/reward_score/__init__.py#L50

import torch
import sys
from pathlib import Path
from tqdm import tqdm

# 添加 slime 到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from slime.utils.types import Sample

# 导入 hanqing 的 reward functions
from hanqing_reward_function import compute_score

# 写死的 rollout 文件夹路径
ROLLOUT_FOLDER_PATH = "/ssd2/zhizhou/workspace/rotation-project/shared_folder/Llama-3.2-3B-Instruct-fsdp-1116-noref/dump_details/rollout_data"


def evaluate_single_rollout_math_score(rollout_file_path):
    """
    评估单个 rollout 文件的 math score
    
    Args:
        rollout_file_path: rollout .pt 文件的路径
        
    Returns:
        dict: 包含评估结果的字典
    """
    # 加载数据
    data = torch.load(rollout_file_path)
    rollout_id = data['rollout_id']
    sample_dicts = data['samples']
    
    # 将字典转换为 Sample 对象
    samples = [Sample.from_dict(s) for s in sample_dicts]
    
    # 评估每个样本的 math score
    results = []
    correct_count = 0
    
    for i, sample in enumerate(samples):
        # 获取 response 和 label
        response = sample.response
        label = sample.label
        
        # 使用 compute_score 函数评估
        # is_longcot=False, is_use_math_verify=True
        try:
            score_result = compute_score(
                solution_str=response,
                ground_truth=label,
                is_longcot=False,
                is_use_math_verify=True
            )
            
            # score_result 应该返回一个分数（1 表示正确，0 表示错误）
            is_correct = score_result == 1 or score_result is True
            
            if is_correct:
                correct_count += 1
            
            results.append({
                'index': i,
                'sample_index': sample.index,
                'response': response[:200] + '...' if len(response) > 200 else response,
                'label': label,
                'score': score_result,
                'is_correct': is_correct,
            })
            
        except Exception as e:
            results.append({
                'index': i,
                'sample_index': sample.index,
                'response': response[:200] + '...' if len(response) > 200 else response,
                'label': label,
                'score': 0,
                'is_correct': False,
                'error': str(e),
            })
    
    # 计算统计信息
    total_samples = len(samples)
    accuracy = correct_count / total_samples if total_samples > 0 else 0
    
    # 打印结果
    print(f"\n{'='*80}")
    print("评估结果")
    print(f"{'='*80}")
    print(f"Rollout file basename: {Path(rollout_file_path).name}")
    print(f"总样本数: {total_samples}")
    print(f"正确数量: {correct_count}")
    print(f"准确率: {accuracy:.4f} ({correct_count}/{total_samples})")
    
    return {
        'rollout_id': rollout_id,
        'filename': Path(rollout_file_path).name,
        'total_samples': total_samples,
        'correct_count': correct_count,
        'accuracy': accuracy,
        'detailed_results': results,
    }


def main():
    # 获取文件夹中所有的 .pt 文件
    import os
    rollout_folder = Path(ROLLOUT_FOLDER_PATH)
    
    if not rollout_folder.exists():
        print(f"错误: 文件夹不存在: {rollout_folder}")
        sys.exit(1)
    
    # 获取并排序所有 .pt 文件
    def get_sort_key(filename):
        name = filename.stem
        if name.startswith('eval_'):
            name = name[5:]
        try:
            return (0, int(name))
        except ValueError:
            return (1, name)
    
    pt_files = sorted(rollout_folder.glob('*.pt'), key=get_sort_key)
    
    if not pt_files:
        print(f"错误: 文件夹中没有找到 .pt 文件: {rollout_folder}")
        sys.exit(1)
    
    # 评估所有文件
    all_results = []
    for pt_file in pt_files:
        result = evaluate_single_rollout_math_score(pt_file)
        all_results.append(result)


if __name__ == "__main__":
    main()










