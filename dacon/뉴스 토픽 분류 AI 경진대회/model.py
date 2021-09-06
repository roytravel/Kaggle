# -*- coding:utf-8 -*-
import os
import re
import numpy as np # calculate matrix
import pandas as pd # control dataframe
import warnings # remove warning log
import matplotlib.pyplot as plt # visualize graph
import tensorflow as tf
from tqdm import tqdm
from transformers import BertModel, TFBertModel, BertTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from dacon_submit_api import dacon_submit_api
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings(action='ignore')


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string], '')
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()


def load_dataset(IN_PATH):
    train_data = pd.read_csv(IN_PATH + "train_data.csv")
    test_data = pd.read_csv(IN_PATH + "test_data.csv")
    submission = pd.read_csv(IN_PATH + "sample_submission.csv")
    return train_data, test_data, submission


def bert_tokenizer(sent, MAX_LEN):
    encoded_dict = tokenizer.encode_plus(
        text = sent,
        add_special_tokens = True, # Add [CLS] & [SEP]
        max_length = MAX_LEN, # Pad & truncate all sentences.
        pad_to_max_length = True,
        return_attention_mask = True) # Construct attn. masks.

    input_id = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask'] # And its attention mask (simply differentiates padding from non-padding).
    token_type_id = encoded_dict['token_type_ids'] # differentiate two sentences
    return input_id, attention_mask, token_type_id


def create_checkpoint_dir(OUT_PATH):
    checkpoint_path = os.path.join(OUT_PATH, model_name, 'weights.h5')
    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    # Create path if exists
    if os.path.exists(checkpoint_dir):
        print("[!] Folder already exists : {}\n".format(checkpoint_dir))
    else:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print("[!] Folder create complete : {}\n".format(checkpoint_dir))

    return checkpoint_dir


class TFBertClassifier(tf.keras.Model):
    def __init__(self, model_name, dir_path, num_class):
        super(TFBertClassifier, self).__init__()

        self.bert = TFBertModel.from_pretrained(model_name, cache_dir=dir_path)
        self.dropout = tf.keras.layers.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(num_class, kernel_initializer=tf.keras.initializers.TruncatedNormal(self.bert.config.initializer_range), name="classifier")
    
    def call(self, inputs, attention_mask=None, token_type_ids=None, training=False):
        #outputs 값: # sequence_output, pooled_output, (hidden_states), (attentions)
        outputs = self.bert(inputs, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1] 
        pooled_output = self.dropout(pooled_output, training=training)
        logits = self.classifier(pooled_output)
        return logits


def create_model():
    model = TFBertClassifier(model_name='bert-base-multilingual-cased', dir_path='exclude/cache', num_class=7)
    optimizer = tf.keras.optimizers.Adam(3e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    return model

# 05. Get sent_label_from_train
def get_sent_label_from_dataset(train_data, test_data, MAX_LEN, prefix):
    # 04. define elements needed in BERT
    input_ids = list()
    attention_masks = list()
    token_type_ids = list()
    train_data_labels = list()
    test_data_labels = list()

    if prefix == "train":
        for train_sent, train_label in tqdm(zip(train_data["title"], train_data["topic_idx"]), total=len(train_data)):
            try:
                input_id, attention_mask, token_type_id = bert_tokenizer(train_sent, MAX_LEN)
                input_ids.append(input_id)
                attention_masks.append(attention_mask)
                token_type_ids.append(token_type_id)
                train_data_labels.append(train_label)

            except Exception as ex:
                print("[*] Exception : {}".format(ex))
                print("[!] {}".format(train_sent))
                pass

        train_movie_input_ids = np.array(input_ids, dtype=int)
        train_movie_attention_masks = np.array(attention_masks, dtype=int)
        train_movie_type_ids = np.array(token_type_ids, dtype=int)
        train_movie_inputs = (train_movie_input_ids, train_movie_attention_masks, train_movie_type_ids)
        train_data_labels = np.asarray(train_data_labels, dtype=np.int32) #레이블 토크나이징 리스트
        print("[*] Sentence: {} / Label: {}".format(len(train_movie_input_ids), len(train_data_labels)))
        return train_movie_inputs, train_data_labels

    if prefix == "test":
        #for test_sent in tqdm(zip(test_data["title"])):
        for test_sent in tqdm(test_data["title"]):
            try:
                input_id, attention_mask, token_type_id = bert_tokenizer(test_sent, MAX_LEN)
                input_ids.append(input_id)
                attention_masks.append(attention_mask)
                token_type_ids.append(token_type_id)
            except Exception as ex:
                print('[*] Error: {}'.format(ex))
                print('[!] {}'.format(test_sent))
                pass

        test_movie_input_ids = np.array(input_ids, dtype=int)
        test_movie_attention_masks = np.array(attention_masks, dtype=int)
        test_movie_type_ids = np.array(token_type_ids, dtype=int)
        test_movie_inputs = (test_movie_input_ids, test_movie_attention_masks, test_movie_type_ids)
        print("[*] Sentence: {} / Label: {}".format(len(test_movie_input_ids), len(test_data_labels)))
        results = model.predict(test_movie_inputs, batch_size=1024)
        return results

def create_submission(submission, results):
    #results=tf.argmax(results, axis=1)
    topic = []
    for i in range(len(results)):
        topic.append(np.argmax(results[i]))
    submission['topic_idx'] = topic
    submission.to_csv('dataset/output/bert_baseline.csv', index=False)
    print ('[*] Finished creating submission file')


def get_score():
    result = dacon_submit_api.post_submission_file(
        'dataset/output/bert_baseline.csv', 
        '64ee6893a79be52e4828fd38b645a0b1429110eecfc3e6026bc50739016c354c', 
        'roytravel', 
        'roytravel', 
        '')
    return result


if __name__ == "__main__":

    # 01. set the hyper-parameter 
    tf.random.set_seed(1234)
    np.random.seed(1234)

    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    VALID_SPLIT = 0.2
    MAX_LEN = 44 # You can get max length using excel function like this --> MAX(LEN(B2:B45655))

    # 02. load dataset    
    IN_PATH = 'C:/github/ai-contest/dacon/뉴스 토픽 분류 AI 경진대회/dataset/input/'
    OUT_PATH = "C:/github/ai-contest/dacon/뉴스 토픽 분류 AI 경진대회/dataset/output/"
    train_data, test_data, submission = load_dataset(IN_PATH)

    # 03. load pretrained tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", cache_dir='exclude/cache', do_lower_case=False)

    # 04. Get sentence and label from train dataset    
    train_movie_inputs, train_data_labels = get_sent_label_from_dataset(train_data, test_data, MAX_LEN, 'train')

    # 05. create model
    model = create_model()
    model_name = "tf2_bert_naver_movie"
    
    # 06. set the callback
    earlystop_callback = EarlyStopping(monitor='val_accuracy', min_delta=0.0001,patience=3) # min_delta: accuracy should at least improve 0.0001
    checkpoint_path = create_checkpoint_dir(OUT_PATH)
    cp_callback = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True)

    # 07. training and evaluation
    history = model.fit(train_movie_inputs, train_data_labels, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_split = VALID_SPLIT, callbacks=[earlystop_callback, cp_callback])
    model.save('C:/github/ai-contest/dacon/뉴스 토픽 분류 AI 경진대회/dataset/output/model.h5')
    #print(history.history) --> 저장 모듈로 바꿀 것

    # 08. Visualize the history as graph
    plot_graphs(history, 'loss')

    # 09. create submission using result by predicting test data
    results = get_sent_label_from_dataset(train_data, test_data, MAX_LEN, 'test')

    create_submission(submission, results)

    score = get_score()

    print ('[*] Score : {}'.format(score))