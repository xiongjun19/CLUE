#!/usr/bin/env bash

set -e
set -x
TASK_NAME="afqmc"
model_name_arr=("voidful/albert_chinese_large" "voidful/albert_chinese_xlarge")
out_suf_arr=("albert_chinese_large" "albert_chinese_xlarge")
model_type_arr=("albert" "albert")
lr_arr=(1e-5 2e-6)


for(( i=0;i<${#model_name_arr[@]};i++)) do
	bash run_classifier_afqmc_albert_general.sh ${model_name_arr[i]} ${out_suf_arr[i]} ${model_type_arr[i]} ${lr_arr[i]};
done


