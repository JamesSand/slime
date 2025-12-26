#!/bin/bash

# Zhizhou: modify from run-qwen3-4B-fsdp.sh

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16
export CUDA_VISIBLE_DEVICES=0,1,2,3

source .env

num_gpus=4
hf_model_folder=/root/shared_folder/hf_models
model_name=Qwen3-0.6B

NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"



SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

RUN_ID=${RUN_ID:-"run_$(date +%Y%m%d_%H%M%S)"}
LOAD_SAVE_PATH="/root/shared_data/${RUN_ID}/checkpoints"

CKPT_ARGS=(
   --hf-checkpoint $hf_model_folder/$model_name
   --load $hf_model_folder/$model_name
   --ref-load $hf_model_folder/$model_name
   # 训练过程中模型的保存路径
   --save /root/shared_data/$model_name-test-1226/
   # 模型保存间隔（步数）
   --save-interval 20
)

EVAL_ARGS=(
   # 评估间隔（Rollout 数）
   --eval-interval 5
   # 评估用的 Prompt 数据集
   --eval-prompt-data aime $hf_model_folder/aime-2024/aime-2024.jsonl
   # 每个评估 Prompt 的采样数量
   --n-samples-per-eval-prompt 16
   # 评估时最大响应长度
   --eval-max-response-len 16384
   # 评估时采样参数
   --eval-top-p 0.7
)

ROLLOUT_ARGS=(
   --prompt-data $hf_model_folder/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --balance-data
   --rm-type deepscaler
   --num-rollout 100
   --rollout-batch-size 8
   --n-samples-per-prompt 8
   --rollout-max-response-len 4096
   --rollout-temperature 0.8
   --global-batch-size 64
)

GRPO_ARGS=(
   --use-kl-loss
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-dev-mcore-fsdp
   --wandb-group qwen3-4B-fsdp-1130-ref
   --wandb-key ${WANDB_API_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.5
   --sglang-decode-log-interval 1000
   --sglang-chunked-prefill-size 4096
   --sglang-attention-backend fa3
)

TRAIN_BACKEND_ARGS=(
   --train-backend fsdp
   --update-weight-buffer-size 536870912
   --gradient-checkpointing
   --attn-implementation flash_attention_3
   --train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}'
)

PERF_ARGS=(
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

MISC_ARGS=(
   --actor-num-nodes 1
   --actor-num-gpus-per-node $num_gpus
   --colocate
   --use-fault-tolerance
   --dump-details /root/shared_data/$model_name-fsdp-1116-noref/dump_details
   # --fsdp-cpu-offload
)

# launch the master node of ray in container - 8 GPUs for training
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus $num_gpus --disable-usage-stats


RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
  }
}"


ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   ${CKPT_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${TRAIN_BACKEND_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${MISC_ARGS[@]}



