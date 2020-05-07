#!/usr/bin/env bash
BATCH=20
TEST_BATCH=5
VOCAB=1050
MLEN=1000
DATA="game"
FIXED_EPOCHS=100
NUM_EPOCHS=101
NHIDDEN=128
PRETRAINED=""
LAST_EPOCH=9
SAMPLE_TIME=1
LEARNING_RATE=0.01
RANK=10
DURATION_SIZE=1000
DURATION_MAX_SIZE=100000
STAND=1
FLAGS="floatX=float32,device=gpu0,dnn.include_path=/home/vallzey/local/cuda-8.0/include,dnn.library_path=/home/vallzey/local/cuda-8.0/lib64"
THEANO_FLAGS="${FLAGS}" python vmain_pos.py --model RNN --data ${DATA} --batch_size ${BATCH} --vocab_size ${VOCAB} --max_len ${MLEN} --fixed_epochs ${FIXED_EPOCHS} --num_epochs ${NUM_EPOCHS} --num_hidden ${NHIDDEN} --test_batch ${TEST_BATCH} --learning_rate ${LEARNING_RATE} --sample_time ${SAMPLE_TIME} --rank ${RANK} --duration_size ${DURATION_SIZE}  --duration_max_size ${DURATION_MAX_SIZE} --stand=${STAND}
