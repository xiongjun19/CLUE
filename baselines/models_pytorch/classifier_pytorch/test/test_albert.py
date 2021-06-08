# coding=utf8


import torch
# from transformers import *
# from transformers import BertTokenizer, AlbertForMaskedLM
from transformers import  AlbertForMaskedLM
# from transformers import *
from torch.nn import functional as F

from transformers import (WEIGHTS_NAME, BertConfig,
                          BertForSequenceClassification, BertTokenizer,
                          RobertaConfig, XLNetConfig,
                          XLNetForSequenceClassification,
                          XLNetTokenizer,
                          AlbertForSequenceClassification)

import transformers
print(f"version is  {transformers.__version__}")
print(f"bert tokenizer {BertTokenizer}")

pretrained = "voidful/albert_chinese_large"
tokenizer = BertTokenizer.from_pretrained(pretrained)
model = AlbertForMaskedLM.from_pretrained(pretrained)

inputtext = "今天[MASK]情很好"

maskpos = tokenizer.encode(inputtext, add_special_tokens=True).index(103)
input_ids = torch.tensor(tokenizer.encode(inputtext, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
# outputs = model(input_ids, masked_lm_labels=input_ids)
outputs = model(input_ids)
prediction_scores = outputs["logits"]
logit_prob = F.softmax(prediction_scores[0, maskpos]).data.tolist()
predicted_index = torch.argmax(prediction_scores[0, maskpos]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
print(predicted_token,logit_prob[predicted_index])

