#!/usr/bin/env python3
"""
Custom reward function for slime using hanqing's compute_score

这个文件可以通过 --custom-rm-path 参数在 slime 中使用

用法:
    --custom-rm-path slime.szz_utils.hanqing_custom_rm.hanqing_reward_function
"""

from hanqing_reward_function import compute_score
from slime.utils.types import Sample


async def hanqing_reward_function(args, sample: Sample, **kwargs):
    """
    使用 hanqing 的 compute_score 作为 reward function
    
    Args:
        args: slime 的参数对象
        sample: Sample 对象，包含 response 和 label
        **kwargs: 其他可选参数
        
    Returns:
        int or float: 奖励分数 (1 表示正确，0 表示错误)
    """
    response = sample.response
    label = sample.label
    
    try:
        # 调用 hanqing 的 compute_score 函数
        # is_longcot=False, is_use_math_verify=True
        score_result = compute_score(
            solution_str=response,
            ground_truth=label,
            is_longcot=False,
            is_use_math_verify=True
        )
        
        return score_result
        
    except Exception as e:
        # 如果评估失败，返回 0
        print(f"Warning: Failed to compute score for sample {sample.index}: {e}")
        return 0
