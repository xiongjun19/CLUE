#!/usr/bin/env bash

set -e
set -x
TASK_NAME="afqmc"
model_name_arr=("voidful/albert_chinese_xlarge" "hfl/chinese-roberta-wwm-ext-large")
out_suf_arr=("albert_chinese_xlarge" "roberta_large")
model_type_arr=("albert" "roberta")


for(( i=0;i<${#model_name_arr[@]};i++)) do
	bash run_classifier_afqmc_general.sh ${model_name_arr[i]} ${out_suf_arr[i]} ${model_type_arr[i]};
done


