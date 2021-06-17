#!/usr/bin/env bash

set -e
set -x
TASK_NAME="afqmc"
model_name_arr=("bert-base-chinese" "voidful/albert_chinese_base" "hfl/chinese-roberta-wwm-ext" "hfl/chinese-roberta-wwm-ext-large")
out_suf_arr=("bert-base-chinese" "albert_chinese" "roberta_large" "roberta_large")
model_type_arr=("bert" "albert" "roberta" "roberta")


for(( i=0;i<${#model_name_arr[@]};i++)) do
	bash run_classifier_afqmc_general.sh ${model_name_arr[i]} ${out_suf_arr[i]} ${model_type_arr[i]};
done


