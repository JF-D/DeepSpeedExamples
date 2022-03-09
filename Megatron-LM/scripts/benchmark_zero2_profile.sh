#! /bin/bash

# Change for multinode config
MP_SIZE=1

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=8

script_path=$(realpath $0)
script_dir=$(dirname $script_path)

nlayer=12
seq_length=1024
hidden_size=1024
nheads=16

bs=1

checkpoint= #"--checkpoint-activations --deepspeed-activation-checkpointing"

config_json="$script_dir/benchmark_zero2_config.json"
gpt_options=" \
       --model-parallel-size ${MP_SIZE} \
       --num-layers ${nlayer} \
       --hidden-size ${hidden_size} \
       --num-attention-heads ${nheads} \
       --batch-size ${bs} \
       --seq-length ${seq_length} \
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
       $checkpoint --synthetic --timeline \
"

gpt_options="${gpt_options}
               --deepspeed \
               --deepspeed_config ${config_json} \
"


# run_cmd="sudo env \"PATH=$PATH\" CC=/usr/bin/gcc-5 CXX=/usr/bin/g++-5 \
#         $(which deepspeed) --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} \
#         pretrain_gpt2.py $@ ${gpt_options}"

# run_cmd="sudo env \"PATH=$PATH\" CC=/usr/bin/gcc-5 CXX=/usr/bin/g++-5 \
#         $(which nvprof) --profile-from-start off --profile-child-processes -o log/gpt_%p.nvvp -f \
#         $(which deepspeed) --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} \
#         pretrain_gpt2.py $@ ${gpt_options}"

# "/usr/local/cuda/bin/nvprof", "--profile-from-start", "off", "-o","log/gpt_%p.nvvp", "-f",

run_cmd="sudo env \"PATH=$PATH\" CC=/usr/bin/gcc-5 CXX=/usr/bin/g++-5 \
        mpirun -np 8 -allow-run-as-root $(which nvprof) --profile-from-start off -o log/gpt_%p.nvvp -f \
        /home/duanjiangfei/env/miniconda3/envs/pt1.8v1/bin/python pretrain_gpt2.py $@ ${gpt_options} --launch mpirun"
echo ${run_cmd}
eval ${run_cmd}

set +x
