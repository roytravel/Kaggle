# -*- coding:utf-8 -*-
import numpy as np
from transformers import BertModel, TFBertModel, BertTokenizer
from model import bert_tokenizer
from tensorflow.keras.models import load_model

#checkpoint_path = "C:\github/ai-contest/dacon/뉴스 토픽 분류 AI 경진대회/dataset/output/"
#checkpoint_path = "dataset/output/checkpoint"
model = load_model(checkpoint_path)

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", cache_dir='exclude/cache', do_lower_case=False)

input_ids = list()
attention_masks = list()
token_type_ids = list()
train_data_labels = list()
test_data_labels = list()

input_id, attention_mask, token_type_id = bert_tokenizer(test_sent, MAX_LEN)
input_ids.append(input_id)
attention_masks.append(attention_mask)
token_type_ids.append(token_type_id)

test_movie_input_ids = np.array(input_ids, dtype=int)
test_movie_attention_masks = np.array(attention_masks, dtype=int)
test_movie_type_ids = np.array(token_type_ids, dtype=int)
test_movie_inputs = (test_movie_input_ids, test_movie_attention_masks, test_movie_type_ids)
print("[*] Sentence: {} / Label: {}".format(len(test_movie_input_ids), len(test_data_labels)))
results = model.predict(test_movie_inputs, batch_size=1024)
