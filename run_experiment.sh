# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Copyright Tor Vergata, University of Rome. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Experiment runner script

set -o nounset
set -o errexit

TASK_NAME=${1}                  # qc-fine
cur_dir="data/${TASK_NAME}"
BERT_BASE_DIR="embeddings/${2}" # cased_L-12_H-768_A-12
SEQ_LEN="64"
BS="64"
LR="2e-5"
EPOCHS="3"
LABEL_RATE=${3}                 # 0.02

rm -rf bert_output_model ganbert_output_model

function ganbert {
python -u ganbert.py \
        --task_name=${TASK_NAME} \
        --label_rate=${LABEL_RATE} \
        --do_train=true \
        --do_eval=true \
        --do_predict=false \
        --data_dir=${cur_dir} \
        --vocab_file=$BERT_BASE_DIR/vocab.txt \
        --bert_config_file=$BERT_BASE_DIR/bert_config.json \
        --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
        --max_seq_length=${SEQ_LEN} \
        --train_batch_size=${BS} \
        --learning_rate=${LR} \
        --num_train_epochs=${EPOCHS} \
        --warmup_proportion=0.1 \
        --do_lower_case=false \
        --output_dir=ganbert_output_model

}

function bert {
python -u bert.py \
        --task_name=${TASK_NAME} \
        --label_rate=${LABEL_RATE} \
        --do_train=true \
        --do_eval=true \
        --do_predict=false \
        --data_dir=${cur_dir} \
        --vocab_file=$BERT_BASE_DIR/vocab.txt \
        --bert_config_file=$BERT_BASE_DIR/bert_config.json \
        --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
        --max_seq_length=${SEQ_LEN} \
        --train_batch_size=${BS} \
        --learning_rate=${LR} \
        --num_train_epochs=${EPOCHS} \
        --warmup_proportion=0.1 \
        --do_lower_case=false \
        --output_dir=bert_output_model
}

ganbert
bert
