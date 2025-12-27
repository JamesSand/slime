import torch
import os
from pprint import pprint

rollout_data_folder = "/ssd2/zhizhou/workspace/rotation-project/shared_folder/Llama-3.2-3B-Instruct-fsdp-1116-noref/dump_details/rollout_data"

# 获取所有 .pt 文件并排序
def get_sort_key(filename):
    name = filename.split('.')[0]
    try:
        return (0, int(name))  # 数字文件优先
    except ValueError:
        return (1, name)  # 字符串文件其次

rollout_files = sorted([f for f in os.listdir(rollout_data_folder) if f.endswith('.pt')], 
                       key=get_sort_key)

print(f"找到 {len(rollout_files)} 个 rollout 文件")
print(f"前几个文件: {rollout_files[:5]}")
print("\n" + "="*80 + "\n")

# 读取前几个文件
num_files_to_read = 3
num_samples_per_file = 2

for i, filename in enumerate(rollout_files[:num_files_to_read]):
    filepath = os.path.join(rollout_data_folder, filename)
    print(f"{'='*80}")
    print(f"读取文件 [{i+1}/{num_files_to_read}]: {filename}")
    print(f"{'='*80}")
    
    # 加载数据
    data = torch.load(filepath)
    
    print(f"\n文件结构:")
    print(f"  - rollout_id: {data['rollout_id']}")
    print(f"  - samples 数量: {len(data['samples'])}")
    
    breakpoint()
    continue
    
    # 打印前几个样本
    print(f"\n前 {min(num_samples_per_file, len(data['samples']))} 个样本:\n")
    
    for j, sample in enumerate(data['samples'][:num_samples_per_file]):
        print("=" * 50)
        print(f"  样本 {j+1}:")
        print(f"    - group_index: {sample['group_index']}")
        print(f"    - index: {sample['index']}")
        print(f"    - prompt: {sample['prompt']}")
        print(f"    - response: {sample['response']}")
        print(f"    - response_length: {sample['response_length']}")
        print(f"    - label: {sample['label']}")
        print(f"    - reward: {sample['reward']}")
        print(f"    - status: {sample['status']}")
        print(f"    - remove_sample: {sample['remove_sample']}")
        
        # 打印 metadata
        if 'metadata' in sample and sample['metadata']:
            print(f"    - metadata 键: {list(sample['metadata'].keys())}")
        
        print()
    
    print("\n")
