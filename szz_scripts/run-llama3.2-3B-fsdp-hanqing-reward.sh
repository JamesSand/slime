#!/bin/bash

# Zhizhou: modify from run-qwen3-4B-fsdp.sh

# Zhizhou: not finished. Inspect parameters:
# num samples, train batch size, temperatures

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

# 函数：寻找一个随机的空闲端口
get_free_port() {
    python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'
}

# 为 Ray 和 Dashboard 自动分配端口
RAY_PORT=$(get_free_port)
RAY_DASHBOARD_PORT=$(get_free_port)

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16
# export CUDA_VISIBLE_DEVICES=0,1
# export CUDA_VISIBLE_DEVICES=2,3


source .env

num_gpus=2
hf_model_folder=/root/shared_folder/hf_models
model_name=Llama-3.2-3B-Instruct

# Zhizhou: I think this is the wandb run name
wandb_group=llama3.2-3B-fsdp-math-adam-hanqing-reward

base_save_folder=/root/shared_folder/$wandb_group
ckpt_save_folder=$base_save_folder/checkpoints
dump_info_save_folder=$base_save_folder/dump_details

mkdir -p $ckpt_save_folder
mkdir -p $dump_info_save_folder

max_response_len=$((1024 * 4))


NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"



SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

RUN_ID=${RUN_ID:-"run_$(date +%Y%m%d_%H%M%S)"}
LOAD_SAVE_PATH="/root/shared_folder/${RUN_ID}/checkpoints"

CKPT_ARGS=(
   --hf-checkpoint $hf_model_folder/$model_name
   --load $hf_model_folder/$model_name
   --ref-load $hf_model_folder/$model_name
   # 训练过程中模型的保存路径
   --save $ckpt_save_folder
   # 模型保存间隔（步数）
   --save-interval 10
)

EVAL_ARGS=(
   # 评估间隔（Rollout 数）
   --eval-interval 5
   # 评估用的 Prompt 数据集
   # --eval-prompt-data aime $hf_model_folder/aime-2024/aime-2024.jsonl
   --eval-prompt-data math500 /root/shared_folder/math_datasets/test_math500/test.jsonl gsm8k /root/shared_folder/math_datasets/test_gsm8k/test.jsonl
   # 每个评估 Prompt 的采样数量
   --n-samples-per-eval-prompt 4
   # 评估时最大响应长度
   --eval-max-response-len $max_response_len
   # 评估时采样参数
   --eval-top-p 0.7
)

# hanqing rollout args
# --custom-rm-path szz_utils.hanqing_custom_rm.hanqing_reward_function

ROLLOUT_ARGS=(
   --prompt-data /root/shared_folder/math_datasets/train_math/train.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --balance-data
   # --rm-type deepscaler
   --custom-rm-path szz_utils.hanqing_custom_rm.hanqing_reward_function
   --num-rollout 100
   --rollout-batch-size 8
   --n-samples-per-prompt 8
   --rollout-max-response-len $max_response_len
   --rollout-temperature 1.0
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
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.999
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
   --dump-details $dump_info_save_folder
   # --fsdp-cpu-offload
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-dev-mcore-fsdp
   --wandb-group $wandb_group
   --wandb-key ${WANDB_API_KEY}
)

# launch the master node of ray in container - 8 GPUs for training
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head \
    --node-ip-address ${MASTER_ADDR} \
    --port ${RAY_PORT} \
    --dashboard-port ${RAY_DASHBOARD_PORT} \
    --num-gpus $num_gpus \
    --disable-usage-stats


RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
  }
}"

LOG_FILE="/root/shared_folder/logs/${model_name}-$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: $LOG_FILE"

ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
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
   ${MISC_ARGS[@]} 2>&1 | tee "$LOG_FILE" 



