#!/usr/bin/env bash
#SBATCH --job-name=ml2023 # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks-per-node=1 # total number of tasks across all nodes
#SBATCH --cpus-per-task=4 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=20G # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:hgx:1 # number of gpus per node
#SBATCH -p pol # -preempted

#SBATCH -o %x-%j.log # output and error log file names (%x for job id)

ROOT_DIR=./
export TORCH_EXTENSIONS_DIR=${ROOT_DIR}/torch_extendsions

MODEL_NAME=erlangshen1.3B
MODEL_ROOT_DIR=$ROOT_DIR/${MODEL_NAME}
if [ ! -d ${MODEL_ROOT_DIR} ];then
  mkdir -p ${MODEL_ROOT_DIR}
fi

NNODES=1
GPUS_PER_NODE=1
MICRO_BATCH_SIZE=32

# 如果你不用Deepspeed的话 下面的一段话都可以删掉 Begin
CONFIG_JSON="$MODEL_ROOT_DIR/${MODEL_NAME}.ds_config.json"
ZERO_STAGE=1

cat <<EOT > $CONFIG_JSON
{
    "zero_optimization": {
        "stage": 1,
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": true
    },
    "fp16": {
        "enabled": false
    },
    "bf16": {
        "enabled": true
    },
    "activation_checkpointing": {
      "partition_activations": true,
      "contiguous_memory_optimization": true,
      "number_checkpoints": 20
    },
    "gradient_clipping": 1,
    "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE
}
EOT
export PL_DEEPSPEED_CONFIG_PATH=$CONFIG_JSON
### End

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=$(shuf -n 1 -i 40000-65535)
# MASTER_PORT=51200
echo $MASTER_ADDR
echo $MASTER_PORT


export LAUNCHER="torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --max_restarts 0 \
    "
CODE_PATH="finetune.py"
DATA_PATH="./data/train.csv"

VALID_DATA_PATH="./data/testA.csv"
MODEL_PATH="../Erlangshen-MegatronBert-1.3B"
OUTPUT_PATH="./output"
PREDICT_FILE="${OUTPUT_PATH}/${MODEL_NAME}.csv"

export WANDB_PROJECT=$MODEL_NAME
export WANDB_LOG_MODEL='all'
export run_name='ml2023'

# --model_name_or_path $MODEL_PATH \

srun --jobid $SLURM_JOBID bash -c `python -m torch.distributed.run \
--nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
--master_addr $MASTER_ADDR --master_port $MASTER_PORT $CODE_PATH \
--data_path $DATA_PATH \
--eval_data_path $VALID_DATA_PATH \
--output_dir $OUTPUT_PATH \
--model_name_or_path $MODEL_PATH \
--model_max_length 512 \
--num_train_epochs 4 \
--per_device_train_batch_size $MICRO_BATCH_SIZE \
--gradient_accumulation_steps 1 \
--learning_rate 1e-5 \
--lr_scheduler_type cosine \
--adam_beta1 0.9 \
--adam_beta2 0.98 \
--adam_epsilon 1e-8 \
--max_grad_norm 1.0 \
--weight_decay 1e-1 \
--warmup_ratio 0.01 \
--logging_steps 1 \
--log_level "debug" \
--bf16 True \
--deepspeed $CONFIG_JSON \
--do_train \
--do_eval \
--evaluation_strategy "steps" \
--save_steps 5000 \
--eval_steps 200 \
--report_to "wandb" \
--run_name $run_name \
--gradient_checkpointing False \
--prediction_file ${PREDICT_FILE} \
`


# 