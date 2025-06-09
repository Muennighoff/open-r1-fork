#!/bin/bash
#SBATCH --job-name=humanline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=XXXXX
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --time 48:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=jobs/%x-%j.out           # output file name

conda activate humanline
echo "START TIME: $(date)"

MODEL=$1
TASK=$2
CONFIG_SUFFIX=$3
ACCELERATOR=$4
OPTIONAL_ARGS=$5
CONFIG_FILE=recipes/$MODEL/$TASK/config_$CONFIG_SUFFIX.yaml
GRAD_ACC_STEPS=$(grep 'gradient_accumulation_steps' $CONFIG_FILE | awk '{print $2}')
MODEL=$(grep 'model_name_or_path:' $CONFIG_FILE | awk '{print $2}')
REVISION=$(grep 'model_revision:' $CONFIG_FILE | head -n 1 | awk '{print $2}')

# Distributed configuration
NUM_NODES=$SLURM_NNODES
GPUS_PER_NODE=8
WORLD_SIZE=$(($NUM_NODES*$GPUS_PER_NODE))
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
MASTER_ADDR=${NODELIST[0]}  # First node for main process
MASTER_PORT=6000

USE_VLLM="false"
if [[ -f "$CONFIG_FILE" ]] && grep -qE '^\s*use_vllm:\s*true' "$CONFIG_FILE"; then
    USE_VLLM="true"
fi

if [[ "$USE_VLLM" == "true" ]]; then
    TP=1
    WORLD_SIZE=$((WORLD_SIZE - TP))
    # set cuda visible devices to first tp gpus
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((TP-1)))
    trl vllm-serve --model $MODEL --revision $REVISION --tensor_parallel_size $TP &
    sleep 300
    OPTIONAL_ARGS="$OPTIONAL_ARGS --vllm_server_host=$MASTER_ADDR"
    # set cuda visible devices to other gpus
    export CUDA_VISIBLE_DEVICES=$(seq -s, $TP $((GPUS_PER_NODE-1)))
fi

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_SOCKET_NTHREADS=1
# export NCCL_NSOCKS_PERTHREAD=1
# export CUDA_LAUNCH_BLOCKING=1

export CMD=" \
    src/open_r1/$TASK.py --config $CONFIG_FILE $OPTIONAL_ARGS
    "

# NCCL stability settings
export NCCL_ASYNC_ERROR_HANDLING=1
export CMD="src/open_r1/$TASK.py --config $CONFIG_FILE $OPTIONAL_ARGS"

HF_HUB_ENABLE_HF_TRANSFER=1 ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch \
    --config_file recipes/accelerate_configs/$ACCELERATOR.yaml \
    --gradient_accumulation_steps $GRAD_ACC_STEPS \
    --num_machines 1 \
    --num_processes $WORLD_SIZE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank 0 \
    --rdzv_backend=c10d \
    $CMD

echo "END TIME: $(date)"