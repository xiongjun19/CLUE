#!/usr/bin/env bash
# @Author: bo.shi
# @Date:   2019-11-04 09:56:36
# @Last Modified by:   bo.shi
# @Last Modified time: 2020-01-01 11:34:22
set -e
set -x

TASK_NAME="afqmc"
MODEL_NAME=$1
out_dir_suffix=$2
model_type=$3

CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
export CUDA_VISIBLE_DEVICES="0"
export BERT_PRETRAINED_MODELS_DIR=$CURRENT_DIR/prev_trained_model
export BERT_WWM_DIR=$BERT_PRETRAINED_MODELS_DIR/$MODEL_NAME
export GLUE_DATA_DIR=$CURRENT_DIR/CLUEdatasets

# download and unzip dataset
if [ ! -d $GLUE_DATA_DIR ]; then
  mkdir -p $GLUE_DATA_DIR
  echo "makedir $GLUE_DATA_DIR"
fi
cd $GLUE_DATA_DIR
if [ ! -d $TASK_NAME ]; then
  mkdir $TASK_NAME
  echo "makedir $GLUE_DATA_DIR/$TASK_NAME"
fi
cd $TASK_NAME
if [ ! -f "train.json" ] || [ ! -f "dev.json" ] || [ ! -f "test.json" ]; then
  rm *
  wget https://storage.googleapis.com/cluebenchmark/tasks/afqmc_public.zip
  unzip afqmc_public.zip
  rm afqmc_public.zip
else
  echo "data exists"
fi
echo "Finish download dataset."

output_dir=$CURRENT_DIR/${TASK_NAME}/${out_dir_suffix}

# make output dir
if [ ! -d ${output_dir} ]; then
  mkdir -p ${output_dir}
  echo "makedir ${output_dir}"
fi

# run task
cd $CURRENT_DIR
echo "Start running..."

python run_albert_classifier.py \
      --model_type=${model_type}\
      --model_name_or_path=$MODEL_NAME \
      --task_name=$TASK_NAME \
      --do_train \
      --do_eval \
      --do_lower_case \
      --data_dir=$GLUE_DATA_DIR/${TASK_NAME}/ \
      --max_seq_length=128 \
      --per_gpu_train_batch_size=16 \
      --per_gpu_eval_batch_size=16 \
      --learning_rate=2e-5 \
      --num_train_epochs=8.0 \
      --logging_steps=2146 \
      --save_steps=2146 \
      --output_dir=${output_dir} \
      --overwrite_output_dir \
      --seed=42

