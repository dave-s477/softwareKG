import tensorflow as tf
import gensim
import numpy as np
import time
import random
import json
import pickle
import math
import time
import sys
import argparse
random.seed(42)

from os import listdir, mkdir
from os.path import join, exists

from tensorflow.keras import Input, Model
from tensorflow.keras.initializers import Constant, GlorotUniform
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import TimeDistributed, Dense, Bidirectional, concatenate, Embedding, LSTM, Dropout
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.metrics import Accuracy, Precision, Recall

from sklearn.metrics import confusion_matrix, classification_report
from multiprocessing import Pool

from util.custom_token_encoder import CustomTokenTextEncoder
from util import crf
from util.doc_info import get_doc_dict

print("Working with Tensorflow {}".format(tf.__version__))

LOG_DIR = 'logs'
CHECKPOINT_DIR = 'checkpoints'
BUFFER_SIZE = 50000
BATCH_SIZE = 512
EMBEDDING_SIZE = 200
CHAR_EMBEDDING_SIZE = 50
DROP_OUT_RATE = 0.3
DOWN_SAMPLE = True
DOWN_SAMPLING_RATE = 0.3
NUM_CROSS_VAL = 10
EPOCHS = 10
CHAR_LSTM_SIZE = 25
WORD_LSTM_SIZE = 100

vocabulary_set = pickle.load(open("vocabs/SoSciSoCi_word_voc.p", "rb" ))
character_set = pickle.load(open("vocabs/SoSciSoCi_char_voc.p", "rb" ))
label_set = pickle.load(open("vocabs/SoSciSoCi_label_voc.p", "rb" ))
vocab_size = len(vocabulary_set)
character_size = len(character_set)
label_size = len(label_set)
print("Working with {} words, {} characters and {} target labels.".format(vocab_size, character_size, label_size))
# Add to all datasets because the counts above do not includ padding

text_encoder = CustomTokenTextEncoder.load_from_file('vocabs/text_encoder_with_pad')
character_encoder = CustomTokenTextEncoder.load_from_file('vocabs/character_encoder_with_pad')
label_encoder = CustomTokenTextEncoder.load_from_file('vocabs/label_encoder_no_pad')
vocabulary_padding = text_encoder.encode('<PAD>')[-1]
character_padding = character_encoder.encode('<PAD>')[-1]
label_padding = label_encoder.encode('<PAD>')[-1]

def encode(text_tensor):
    encoded_text = text_encoder.encode(text_tensor.numpy())
    plain_text = text_tensor.numpy().split()
    encoded_char_list = []
    max_length = 0
    for tok in plain_text:
        some_plain_text = tf.compat.as_text(tok)
        target = []
        for x in some_plain_text:
            char = character_encoder.encode(x)
            if char:
                target.append(char[0])
        if len(target) > 0:
            encoded_char_list.append(target) 
        if len(target) > max_length:
            max_length = len(target)
    sentence_length = len(encoded_text)
    tensor = np.full((len(encoded_text), max_length), character_padding, dtype=np.int32)
    for i, ex in enumerate(encoded_char_list):
        tensor[i,0:len(ex)] = ex
    return encoded_text, tensor, sentence_length

def encode_map_fn(text):
    return tf.py_function(encode, inp=[text], Tout=(tf.int32, tf.int32, tf.int32))

def combine_features(text_tensor, plain_tensor, length_tensor):
    return (text_tensor, plain_tensor, length_tensor)

def create_prediction_data(data_file_list, batched=True, batch_size=64):
    lines_dataset = tf.data.TextLineDataset(data_file_list)
    labelled_dataset = tf.data.Dataset.zip((lines_dataset))
    labelled_dataset = labelled_dataset.map(encode_map_fn)
    labelled_dataset = labelled_dataset.map(combine_features)
    if batched:
        labelled_dataset = labelled_dataset.padded_batch(batch_size, padded_shapes=(([None], [None,None], [])),padding_values=((tf.constant(vocabulary_padding, dtype=tf.int32), tf.constant(character_padding, dtype=tf.int32), tf.constant(0, dtype=tf.int32))))#(([-1],[-1,-1]),[-1]))#, padding_values=(0, 0))
    return labelled_dataset

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def get_software_sent(pred_and_text):
    sent_software_list = []
    current_software = ''
    for word_pred, word_text in zip(label_encoder.decode(pred_and_text[0]).split(), text_encoder.decode(pred_and_text[1]).split()):
        if word_text == '<PAD>':
            if current_software:
                sent_software_list.append(current_software)
            break
        if word_pred == 'O':
            if current_software:
                sent_software_list.append(current_software)
                current_software = ''
        elif word_pred == 'B-software':
            if current_software: 
                sent_software_list.append(current_software)
            current_software = word_text
        elif word_pred == 'I-software':
            if current_software:
                current_software += ' ' + word_text
            else:
                current_software = word_text
    return sent_software_list

def get_software_names_multi(pred_seq, text_tensor):
    with Pool(32) as p:
        sent_software_nested = p.map(get_software_sent, zip(pred_seq, text_tensor))
    return [item for sublist in sent_software_nested for item in sublist]

def get_software_names(pred_seq, text_tensor, sent_length):
    batch_software_list = []
    for sent_pred, sent_text, cur_len in zip(pred_seq, text_tensor, sent_length):
        sent_software_list = []
        current_software = ''
        for idx, (word_pred, word_text) in enumerate(zip(label_encoder.decode(sent_pred).split(), text_encoder.decode(sent_text).split())):
            if word_text == '<PAD>':
                if current_software:
                    sent_software_list.append(current_software)
                break
            if word_pred == 'O':
                if current_software:
                    sent_software_list.append(current_software)
                    current_software = ''
            elif word_pred == 'B-software':
                if current_software: 
                    sent_software_list.append(current_software)
                current_software = word_text
            elif word_pred == 'I-software':
                if current_software:
                    current_software += ' ' + word_text
                else:
                    current_software = word_text
        batch_software_list.extend(sent_software_list)
    return batch_software_list

def predict_article(article_name, model):
    line_number = file_len('../data/reasoning_data/{}'.format(article_name))
    article_name_doi = article_name.split('.txt')[0]
    if line_number <= 1:
        print("Found practically empty file.. {}".format(article_name))
        article_info = get_doc_dict(article_name_doi, [])
    else:
        article_set = create_prediction_data(join('../data/reasoning_data', article_name), batched=True, batch_size=max(BATCH_SIZE, line_number))
        software_names = []
        for b in article_set:
            pred_seq = model(b, train=False)
            if line_number < 70:
                extracted_software = get_software_names(pred_seq, b[0], sent_length)
            else:
                extracted_software = get_software_names_multi(pred_seq, b[0])
            software_names.extend(extracted_software)
        article_info = get_doc_dict(article_name_doi, software_names)
    return article_info

class bi_LSTM_seq_tagger(tf.keras.models.Model):
    def __init__(self, initializer, embedding_layer, drop_rate, char_num, char_embedding_size, char_lstm_size, lstm_size, label_size, **kwargs):
        super().__init__(**kwargs)
        self.drop_rate = drop_rate
        self.char_num = char_num
        self.char_embedding_size = char_embedding_size
        self.char_lstm_size = char_lstm_size
        self.lstm_size = lstm_size
        self.label_size = label_size
        self.word_embedding = embedding_layer
        self.word_emb_dropout = Dropout(drop_rate)
        self.char_embedding = Embedding(char_num, char_embedding_size, trainable=True)
        self.char_emb_dropout = Dropout(drop_rate)
        self.char_lstm = TimeDistributed(Bidirectional(LSTM(char_lstm_size, return_sequences=False)))
        self.char_feature_dropout = Dropout(drop_rate)
        self.feature_lstm = Bidirectional(LSTM(lstm_size, return_sequences=True))
        self.feature_dropout = Dropout(drop_rate)
        self.logits = TimeDistributed(Dense(label_size, activation="relu"))
        self.crf_params = tf.Variable(initializer([label_size, label_size]), "transitions")
 
    def call(self, inputs, training=False, targ_seq=None):
        input_words = inputs[0]
        word_emb = self.word_embedding(input_words)
        if training:
            word_emb = self.word_emb_dropout(word_emb)
        
        input_chars = inputs[1]
        char_emb = self.char_embedding(input_chars)
        if training:
            char_emb = self.char_emb_dropout(char_emb)
        char_features = self.char_lstm(char_emb)
        if training:
            char_features = self.char_feature_dropout(char_features)
        
        sequence_length = inputs[2]
        sequence_length = tf.squeeze(sequence_length)
        features = concatenate([word_emb, char_features])
        lstm_features = self.feature_lstm(features)
        if training:
            lstm_features = self.feature_dropout(lstm_features)
        classification = self.logits(lstm_features)

        if training:
            log_likelihood, _ = crf_log_likelihood(classification, targ_seq, sequence_length, self.crf_params)
            return log_likelihood
        else:
            tag_seq, scores = crf_decode(logits, self.crf_params, sequence_length)
            if targ_seq is not None:
                log_likelihood, _ = crf_log_likelihood(classification, targ_seq, sequence_length, self.crf_params)
                return tag_seq, log_likelihood
            else:  
                return tag_seq       
    
    def get_config(self):
        return {
            "drop_rate": self.drop_rate,
            "char_num": self.char_num,
            "char_embedding_size": self.char_embedding_size,
            "char_lstm_size": self.char_lstm_size,
            "lstm_size": self.lstm_size,
            "label_size": self.label_size
        }

def run_prediction(args, output_location):
    reasoning_files = pickle.load(open("../data/reasoning_set_{}.p".format(args.reasoning_set), "rb"))    

    word_embedding = gensim.models.KeyedVectors.load_word2vec_format(join('embeddings', 'wikipedia-pubmed-and-PMC-w2v.bin'), binary=True)
    embedding = np.random.normal(0.0, 1.0, (vocab_size, EMBEDDING_SIZE))
    found = 0
    not_found = 0
    for word in vocabulary_set:
        if word in word_embedding.vocab:
            found += 1
            word_vector = word_embedding.wv[word]
            embedding[text_encoder.encode(word)[0]-1] = word_vector
        else:
            not_found += 1
    print('Found {} of {} words ({} not found).'.format(found, vocab_size, not_found))
    word_embedding = None

    embedding_layer = Embedding(vocab_size,
                                EMBEDDING_SIZE,
                                embeddings_initializer=Constant(embedding),
                                trainable=False)
    b_software_label = label_encoder.encode('B-software')[0]
    i_software_label = label_encoder.encode('I-software')[0]
    default_label = label_encoder.encode('O')[0]

    initializer = GlorotUniform()
    model = bi_LSTM_seq_tagger(initializer=initializer,
                               embedding_layer=embedding_layer,
                               drop_rate=DROP_OUT_RATE, 
                               char_num=character_size, 
                               char_embedding_size=CHAR_EMBEDDING_SIZE, 
                               char_lstm_size=CHAR_LSTM_SIZE,
                               lstm_size=int(args.lstm_size),
                              # lstm_size=WORD_LSTM_SIZE,
                               label_size=label_size)

    model_location = 'checkpoints/{}/model'.format(args.checkpoint)
    print(model_location)
    model.load_weights(model_location)

    start_time = time.time()
    for idx, test_article in enumerate(reasoning_files):
        if idx % 100 == 0:
            print("At article {}".format(idx))
            print("Total elapsed time {}".format(time.time() - start_time))
        #print(join(output_location, test_article.split('.txt')[0]+'.json'))
        with open(join(output_location, test_article.split('.txt')[0]+'.json'), 'w') as json_file:
            article_dict = predict_article(test_article, model)
            json.dump(article_dict, json_file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Predict for reasoning with a trained Bi-LSTM-CRF model.")
    
    parser.add_argument("--checkpoint", required=False, help="Model weights to use for prediction")
    parser.add_argument("--reasoning-set", required=True, help="reasoning set to be assessed")
    parser.add_argument("--lstm-size", required=True, help="Needs to match the size of the loaded model")
    args = parser.parse_args()

    # Create the output location
    output_location = join('data', 'reasoning_output_{}'.format(args.checkpoint))
    if not exists(output_location):
        mkdir(output_location)

    run_prediction(args, output_location)