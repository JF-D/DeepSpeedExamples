#! /bin/bash

# Runs the "345M" parameter model

nlayer=12
seq_length=1024
hidden_size=1024
nheads=16

ndev=8
bs=1

checkpoint= #"--checkpoint-activations"

vocab_size=40478

mpirun -np $ndev python pretrain_gpt2.py \
       --num-layers $nlayer \
       --hidden-size $hidden_size \
       --num-attention-heads $nheads \
       --batch-size $bs\
       --seq-length $seq_length \
       --max-position-embeddings $seq_length \
       --train-iters 320000 \
       --train-data wikipedia \
       --lazy-loader \
       --tokenizer-type GPT2BPETokenizer \
       --cache-dir cache \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --log-interval 1 \
       --vocab-size $vocab_size \
       --warmup .01 \
       --no-fuse-adam \
       $checkpoint --synthetic \
       --launch mpirun


set +x
