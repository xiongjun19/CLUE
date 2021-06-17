#!/usr/bin/env bash

set -e
set -x
model_name_arr=("bert-base-chinese" "voidful/albert_chinese_base" "voidful/albert_chinese_large" "voidful/albert_chinese_xlarge" "hfl/chinese-roberta-wwm-ext" "hfl/chinese-roberta-wwm-ext-large")
out_suf_arr=("bert-base-chinese" "albert_chinese" "albert_chinese_large" "albert_chinese_xlarge" "roberta_large" "roberta_large")
model_type_arr=("bert" "albert" "albert" "albert" "roberta" "roberta")


for(( i=0;i<${#model_name_arr[@]};i++)) do
	bash run_classifier_iflytek_general.sh ${model_name_arr[i]} ${out_suf_arr[i]} ${model_type_arr[i]};
done


