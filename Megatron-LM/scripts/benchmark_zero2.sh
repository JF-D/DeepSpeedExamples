#! /bin/bash

# Change for multinode config

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=8

script_path=$(realpath $0)
script_dir=$(dirname $script_path)

nlayer=48
seq_length=1024
hidden_size=1600
nheads=16
vocab_size=50257

ndev=$1
PROF=$2

g=$(($ndev<8?$ndev:8))

bs=1
MP_SIZE=1

checkpoint="--checkpoint-activations --deepspeed-activation-checkpointing"

prefix="zero_dp"

config_json="$script_dir/benchmark_zero2_config.json"
gpt_options=" \
       --model-parallel-size ${MP_SIZE} \
       --num-layers ${nlayer} \
       --hidden-size ${hidden_size} \
       --num-attention-heads ${nheads} \
       --batch-size ${bs} \
       --seq-length ${seq_length} \
       --vocab-size ${vocab_size} \
       --max-position-embeddings ${seq_length} \
       --train-iters 100000 \
       --train-data webtext \
       --lazy-loader \
       --tokenizer-type GPT2BPETokenizer \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --no-load-optim \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --no-fuse-adam \
       $checkpoint --synthetic \
"

gpt_options="${gpt_options}
               --deepspeed \
               --deepspeed_config ${config_json} \
"

CMD_PREFIX=""
CMD_SUFFIX=""
if [ "${PROF}" = "profile" ]; then
    export LD_LIBRARY_PATH=/nvme/platform/dep/cuda11.0-cudnn8.0/nsight-systems-2020.3.2/target-linux-x64:$LD_LIBRARY_PATH
    CMD_PREFIX="/nvme/platform/dep/cuda11.0-cudnn8.0/bin/nsys profile -t cudnn,cuda,osrt,nvtx \
                --output log/nvvp/gpt_n${ndev}_%q{SLURM_PROCID} --force-overwrite true"
    CMD_SUFFIX="--timeline"
fi

# run_cmd="deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} pretrain_gpt2.py $@ ${gpt_options}"
# run_cmd="NCCL_DEBUG=INFO /home/duanjiangfei/env/openmpi-2.1.6-cuda-10.1/bin/mpirun -np ${ndev} python -u pretrain_gpt2.py $@ ${gpt_options} --launch mpirun"
srun -p pat_test -x SH-IDC1-10-198-4-[185] -n $ndev --gres=gpu:$g --tasks-per-node=$g \
       ${CMD_PREFIX} python pretrain_gpt2.py ${gpt_options} ${CMD_SUFFIX} 2>&1 | tee log/${prefix}_${ndev}.log
echo ${run_cmd}
eval ${run_cmd}

# MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 WORLD_SIZE=8 RANK=0 LOCAL_RANK=0 python -u pretrain_gpt2.py $@ ${gpt_options} --local_rank 0 --launch nothing &
# MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 WORLD_SIZE=8 RANK=1 LOCAL_RANK=1 python -u pretrain_gpt2.py $@ ${gpt_options} --local_rank 1 --launch nothing &
# MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 WORLD_SIZE=8 RANK=2 LOCAL_RANK=2 python -u pretrain_gpt2.py $@ ${gpt_options} --local_rank 2 --launch nothing &
# MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 WORLD_SIZE=8 RANK=3 LOCAL_RANK=3 python -u pretrain_gpt2.py $@ ${gpt_options} --local_rank 3 --launch nothing &
# MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 WORLD_SIZE=8 RANK=4 LOCAL_RANK=4 python -u pretrain_gpt2.py $@ ${gpt_options} --local_rank 4 --launch nothing &
# MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 WORLD_SIZE=8 RANK=5 LOCAL_RANK=5 python -u pretrain_gpt2.py $@ ${gpt_options} --local_rank 5 --launch nothing &
# MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 WORLD_SIZE=8 RANK=6 LOCAL_RANK=6 python -u pretrain_gpt2.py $@ ${gpt_options} --local_rank 6 --launch nothing &
# MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 WORLD_SIZE=8 RANK=7 LOCAL_RANK=7 python -u pretrain_gpt2.py $@ ${gpt_options} --local_rank 7 --launch nothing &

set +x
