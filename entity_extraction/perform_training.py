import tensorflow as tf
import tensorflow_datasets as tfds
import gensim
import numpy as np
import time
import random
import json
import pickle
import sys
import argparse
random.seed(42)

from os.path import join, exists
from os import mkdir

from tensorflow.keras import Input, Model
from tensorflow.keras.initializers import Constant, GlorotUniform
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import TimeDistributed, Dense, Bidirectional, concatenate, Embedding, LSTM, Dropout
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.metrics import Accuracy, Precision, Recall
from util import crf
from util.eval_fct import evaluate_flat
from util.scores import get_main_scores

from sklearn.metrics import confusion_matrix, classification_report

from util.custom_token_encoder import CustomTokenTextEncoder

print("Working with Tensorflow {}".format(tf.__version__))

# Fixed parameters - including fixed model hyperparameters
LOG_DIR = 'logs'
CHECKPOINT_DIR = 'checkpoints'
BUFFER_SIZE = 50000
BATCH_SIZE = 64
EMBEDDING_SIZE = 200
CHAR_EMBEDDING_SIZE = 50
DOWN_SAMPLE = True
DOWN_SAMPLING_RATE = 0.3
CHAR_LSTM_SIZE = 25
WORD_LSTM_SIZE = 100

# Fixing the datapaths that are available and loading the static vocabularies
silver_standard_positive = 'pos_silver_samples_cor'
silver_standard_negative = 'neg_silver_samples_cor'
merged_silver_standard_1 = 'merged_silver_standard_1'
merged_silver_standard_2 = 'merged_silver_standard_2'
gold_standard_train = 'sosci_train'
gold_standard_devel = 'sosci_devel'
gold_standard_test = 'sosci_test'

vocabulary_set = pickle.load(open("vocabulary_set_with_pad.p", "rb" ))
character_set = pickle.load(open("character_set_with_pad.p", "rb" ))
#label_set = pickle.load(open("label_set_with_pad.p", "rb" ))
label_set = pickle.load(open("label_set_no_pad.p", "rb" ))
vocab_size = len(vocabulary_set)
character_size = len(character_set)
label_size = len(label_set)
print("Working with {} words, {} characters and {} target labels.".format(vocab_size, character_size, label_size))

# Setting up encoders to map words, characters and labels to fixed tf.int32 
text_encoder = CustomTokenTextEncoder.load_from_file('text_encoder_with_pad')
character_encoder = CustomTokenTextEncoder.load_from_file('character_encoder_with_pad')
#label_encoder = CustomTokenTextEncoder.load_from_file('label_encoder_with_pad')
label_encoder = CustomTokenTextEncoder.load_from_file('label_encoder_no_pad')
vocabulary_padding = text_encoder.encode('<PAD>')[-1]
character_padding = character_encoder.encode('<PAD>')[-1]
#label_padding = label_encoder.encode('<PAD>')[-1]
label_padding = label_encoder.encode('O')[-1]

# Data loading and transformation functions
def encode(text_tensor, label_tensor):
    #print(text_tensor)
    encoded_text = text_encoder.encode(text_tensor.numpy())
    plain_text = text_tensor.numpy().split()
    encoded_char_list = []
    max_length = 0
    for tok in plain_text:
        some_plain_text = tf.compat.as_text(tok)
        target = []
        for x in some_plain_text:
            char = character_encoder.encode(x)
            #print(char)
            if char:
                target.append(char[0])
        if len(target) > 0:
            encoded_char_list.append(target) #target = character_encoder.encode('<UNK>') 
        #target = [character_encoder.encode(x)[-1] for x in some_plain_text if len(x) > 0]
        #target = [character_encoder.encode(x)[0] for x in list(tok.decode('utf-8'))]
        if len(target) > max_length:
            max_length = len(target)
    sentence_length = len(encoded_text)
    tensor = np.full((len(encoded_text), max_length), character_padding, dtype=np.int32)
    for i, ex in enumerate(encoded_char_list):
        tensor[i,0:len(ex)] = ex
    encoded_labels = label_encoder.encode(label_tensor.numpy())[0:len(encoded_text)] # This just captures a special case which only concerns samples which are all negativ, so we can just cut of the last labels
    return encoded_text, tensor, sentence_length, encoded_labels

def encode_map_fn(text, label):
    return tf.py_function(encode, inp=[text, label], Tout=(tf.int32, tf.int32, tf.int32, tf.int32))

def combine_features(text_tensor, plain_tensor, length_tensor, label_tensor):
    return (text_tensor, plain_tensor, length_tensor), label_tensor

def downsample(text, plain, seq_length, label):
    existing_labels = set(label.numpy())
    if len(existing_labels) == 1:# and existing_labels[0] == default_label[0]:
        if random.random() < DOWN_SAMPLING_RATE:
            return False
        else:
            return True
    else:
        return True

def downsample_filter_fn(text, plain, seq_length, label):
    return tf.py_function(downsample, inp=[text,plain,seq_length,label], Tout=tf.bool)

def create_dataset(data_file_list, labels_file_list, shuffle=True, down_sample=False, take=None, skip=None):
    lines_dataset = tf.data.TextLineDataset(data_file_list)
    labels_dataset = tf.data.TextLineDataset(labels_file_list)
    labelled_dataset = tf.data.Dataset.zip((lines_dataset, labels_dataset))
    if skip:
        labelled_dataset = labelled_dataset.skip(int(skip))
    if take:
        labelled_dataset = labelled_dataset.take(int(take))
    if shuffle:
        labelled_dataset = labelled_dataset.shuffle(BUFFER_SIZE, reshuffle_each_iteration=True, seed=42)
    labelled_dataset = labelled_dataset.map(encode_map_fn)
    if down_sample:
        labelled_dataset = labelled_dataset.filter(downsample_filter_fn)
    labelled_dataset = labelled_dataset.map(combine_features)
    labelled_dataset = labelled_dataset.padded_batch(BATCH_SIZE, padded_shapes=(([None], [None,None], []),[None]), padding_values=((tf.constant(vocabulary_padding, dtype=tf.int32), tf.constant(character_padding, dtype=tf.int32), tf.constant(0, dtype=tf.int32)), tf.constant(label_padding, dtype=tf.int32)))
    return labelled_dataset

# The model based on the tf.keras subclassing API
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
 
    def call(self, inputs, training=False):
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
        #print("FEATURES")
        #print(features)
        lstm_features = self.feature_lstm(features)
        #print("LSTM Output")
        #print(lstm_features)
        if training:
            lstm_features = self.feature_dropout(lstm_features)
        classification = self.logits(lstm_features)
        tag_seq, score = crf.crf_decode(classification, self.crf_params, sequence_length)
        
        return classification, self.crf_params, sequence_length, tag_seq
    
    def get_config(self):
        return {
            "drop_rate": self.drop_rate,
            "char_num": self.char_num,
            "char_embedding_size": self.char_embedding_size,
            "char_lstm_size": self.char_lstm_size,
            "lstm_size": self.lstm_size,
            "label_size": self.label_size
        }

# Training function performing the training loop 
#@tf.function 
def train_test_func(model, train_set, test_set, train_epochs, learning_rate, learning_decay, global_epochs, save_location, log_location, train_test, train_logging, val_freq, sample_weighting, custom_eval=None):
    # Setup for complete tensorboard logging
    b_software_label = label_encoder.encode('B-software')[0]
    b_software_tensor = tf.constant(b_software_label)
    i_software_label = label_encoder.encode('I-software')[0]
    default_label = label_encoder.encode('O')[0]
    writer = tf.summary.create_file_writer(join(LOG_DIR, log_location))
    checkpoint_name = join(CHECKPOINT_DIR, save_location, 'model')

    # Setting up training and logging
    train_acc_metric = Accuracy()
    if train_logging:
        train_b_soft_recall_metric = Recall()
        train_b_soft_precision_metric = Precision()
        train_i_soft_recall_metric = Recall()
        train_i_soft_precision_metric = Precision()
    val_acc_metric = Accuracy()
    val_b_soft_recall_metric = Recall()
    val_b_soft_precision_metric = Precision()
    val_i_soft_recall_metric = Recall()
    val_i_soft_precision_metric = Precision()
    #val_O_recall_metric = Recall()
    #val_O_precision_metric = Precision()
    val_acc_prev = 0
    val_b_soft_precision_prev = 0
    val_b_soft_recall_prev = 0
    val_i_soft_precision_prev = 0
    val_i_soft_recall_prev = 0
    #val_O_precision_prev = 0
    #val_O_recall_prev = 0    
    
    with writer.as_default():
        for epoch in range(train_epochs):
            learn_rate = learning_rate - learning_decay * epoch 
            print("Current learn rate at {}".format(learn_rate))
            optimizer = RMSprop(learning_rate=learn_rate)
            tf.summary.scalar("Learning Rate", learn_rate, step=global_epochs)
            epoch_start_time = time.time()
            print('Start of epoch %d' % (global_epochs,))

            if train_test == 'train':
                for step, (x_batch_train, y_batch_train) in enumerate(train_set):
                    #if step % 100 == 0:
                    #    print(model.crf_params)
                    with tf.GradientTape() as tape: # watch_accessed_variables=False
                        logits, crf_params, sequence_length, tag_seq = model(x_batch_train, training=True)
                        log_likelihood, _ = crf.crf_log_likelihood(logits, y_batch_train, sequence_length, crf_params)
                        #print(log_likelihood)
                        if sample_weighting:
                            # Up-weigh positive samples while down-weighing negative ones.
                            #print(y_batch_train)
                            #print(tf.math.equal(y_batch_train, b_software_tensor))
                            #print(tf.reduce_any(tf.math.equal(y_batch_train, b_software_tensor), axis=1))
                            #print(tf.where(tf.reduce_any(tf.math.equal(y_batch_train, b_software_tensor), axis=1), 1+float(sample_weighting), 1-float(sample_weighting)))
                            sample_weights = tf.where(tf.reduce_any(tf.math.equal(y_batch_train, b_software_tensor), axis=1), 1+float(sample_weighting), 1-float(sample_weighting))
                            #print(sample_weights)
                            #print(log_likelihood * sample_weights)
                            #print(-log_likelihood * sample_weights)
                            loss = tf.reduce_mean(-log_likelihood * sample_weights)
                        else:
                            loss = tf.reduce_mean(-log_likelihood)
                    grads = tape.gradient(loss, model.trainable_weights)
                    optimizer.apply_gradients(zip(grads, model.trainable_weights))

                    acc_mask = tf.sequence_mask(sequence_length)
                    train_acc_metric(y_batch_train, tag_seq, sample_weight=acc_mask) 
                    if train_logging:
                        y_b_software = tf.where(y_batch_train == b_software_label, 1, 0)
                        pred_b_software = tf.where(tag_seq == b_software_label, 1, 0)
                        train_b_soft_precision_metric(y_b_software, pred_b_software, sample_weight=acc_mask)
                        train_b_soft_recall_metric(y_b_software, pred_b_software, sample_weight=acc_mask)
                        y_i_software = tf.where(y_batch_train == i_software_label, 1, 0)
                        pred_i_software = tf.where(tag_seq == i_software_label, 1, 0)
                        train_i_soft_precision_metric(y_i_software, pred_i_software, sample_weight=acc_mask)
                        train_i_soft_recall_metric(y_i_software, pred_i_software, sample_weight=acc_mask)
                    if step % 20 == 0:
                        print('Training loss (for one batch) at step %s: %s' % (step, float(loss)))
                        print('Seen so far: %s samples' % ((step + 1) * BATCH_SIZE))     

                    #print(model.feature_lstm.weights)
                    #print("variables 0")
                    #print(model.feature_lstm.variables[0])
                    #print("variables 1")
                    #print(model.feature_lstm.variables[1])

                    #print(model.feature_lstm.variables tf.split(z, 4, axis=1)
                    #tf.summary.histogram('LSTM cell', model.feature_lstm)           

                train_acc = train_acc_metric.result()
                print('Training acc over epoch: %s' % (float(train_acc),))
                train_acc_metric.reset_states()
                #tf.summary.scalar("Train Accuracy", train_acc, step=global_epochs)
                tf.summary.scalar("Train Accuracy", train_acc, step=global_epochs)
                if train_logging:
                    train_b_soft_precision = train_b_soft_precision_metric.result()
                    train_b_soft_recall = train_b_soft_recall_metric.result()
                    train_i_soft_precision = train_i_soft_precision_metric.result()
                    train_i_soft_recall = train_i_soft_recall_metric.result()
                    print('train B-software precision over epoch: %s' % (float(train_b_soft_precision),))
                    print('train B-software recall over epoch: %s' % (float(train_b_soft_recall),))
                    print('train I-software precision over epoch: %s' % (float(train_i_soft_precision),))
                    print('train I-software recall over epoch: %s' % (float(train_i_soft_recall),))
                    train_b_soft_precision_metric.reset_states()
                    train_b_soft_recall_metric.reset_states()
                    train_i_soft_precision_metric.reset_states()
                    train_i_soft_recall_metric.reset_states()
                    #tf.summary.scalar("train B-software Precision", train_b_soft_precision, step=global_epochs)
                    #tf.summary.scalar("train B-software Recall", train_b_soft_recall, step=global_epochs)
                    #tf.summary.scalar("train I-software Precision", train_i_soft_precision, step=global_epochs)
                    #tf.summary.scalar("train I-software Recall", train_i_soft_recall, step=global_epochs)
                    tf.summary.scalar("train B-software Precision", train_b_soft_precision, step=global_epochs)
                    tf.summary.scalar("train B-software Recall", train_b_soft_recall, step=global_epochs)
                    tf.summary.scalar("train B-software Fscore", (2 * train_b_soft_precision * train_b_soft_recall) / (train_b_soft_precision + train_b_soft_recall), step=global_epochs)
                    tf.summary.scalar("train I-software Precision", train_i_soft_precision, step=global_epochs)
                    tf.summary.scalar("train I-software Recall", train_i_soft_recall, step=global_epochs)
                    tf.summary.scalar("train I-software Fscore", (2 * train_i_soft_precision * train_i_soft_recall) / (train_i_soft_precision + train_i_soft_recall), step=global_epochs)
                epoch_end_time = time.time()
            
                # Save the model 
                model.save_weights(checkpoint_name, save_format="tf")
                print("Saved trained weights.")
                print("Epoch took %s seconds" % (epoch_end_time - epoch_start_time))

            validation_start_time = time.time()
            print('Start of validation for epoch %d' % (global_epochs,))

            # allows skipping validations but is statically disabled right now
            if epoch % val_freq != 0:
                val_acc_tensorboard = tf.summary.scalar("Validation Accuracy", val_acc_prev, step=global_epochs)
                tf.summary.scalar("B-software Precision", val_b_soft_precision_prev, step=global_epochs)
                tf.summary.scalar("B-software Recall", val_b_soft_precision_prev, step=global_epochs)
                tf.summary.scalar("I-software Precision", val_i_soft_precision_prev, step=global_epochs)
                tf.summary.scalar("I-software Recall", val_i_soft_precision_prev, step=global_epochs)
                #tf.summary.scalar("O Precision", val_O_precision_prev, step=global_epochs)
                #tf.summary.scalar("O Precision", val_O_precision_prev, step=global_epochs)
                writer.flush()
                global_epochs += 1
                continue

            if not custom_eval:
                # Run a validation loop at the end of each epoch.
                for x_batch_val, y_batch_val in test_set:
                    _, _, val_sequence_length, val_tag_seq = model(x_batch_val, training=False)
                    #_, _, val_sequence_length, val_tag_seq = model.predict_on_batch(x_batch_val)
                    
                    val_acc_mask = tf.sequence_mask(val_sequence_length)
                    val_acc_metric(y_batch_val, val_tag_seq, sample_weight=val_acc_mask)

                    #y_b_software = tf.map_fn(lambda x: tf.map_fn(lambda a: 1 if a == b_software_label else 0, x), y_batch_val)
                    #pred_b_software = tf.map_fn(lambda x: tf.map_fn(lambda a: 1 if a == b_software_label else 0, x), val_tag_seq)
                    y_b_software = tf.where(y_batch_val == b_software_label, 1, 0)
                    pred_b_software = tf.where(val_tag_seq == b_software_label, 1, 0)
                    val_b_soft_precision_metric(y_b_software, pred_b_software, sample_weight=val_acc_mask)
                    val_b_soft_recall_metric(y_b_software, pred_b_software, sample_weight=val_acc_mask)

                    #y_i_software = tf.map_fn(lambda x: tf.map_fn(lambda a: 1 if a == i_software_label else 0, x), y_batch_val)
                    #pred_i_software = tf.map_fn(lambda x: tf.map_fn(lambda a: 1 if a == i_software_label else 0, x), val_tag_seq)
                    y_i_software = tf.where(y_batch_val == i_software_label, 1, 0)
                    pred_i_software = tf.where(val_tag_seq == i_software_label, 1, 0)
                    val_i_soft_precision_metric(y_i_software, pred_i_software, sample_weight=val_acc_mask)
                    val_i_soft_recall_metric(y_i_software, pred_i_software, sample_weight=val_acc_mask)

                    #y_O = tf.map_fn(lambda x: tf.map_fn(lambda a: 1 if a == default_label else 0, x), y_batch_val)
                    #pred_O = tf.map_fn(lambda x: tf.map_fn(lambda a: 1 if a == default_label else 0, x), val_tag_seq)
                    #val_O_precision_metric(y_O, pred_O, sample_weight=val_acc_mask)
                    #val_O_recall_metric(y_O, pred_O, sample_weight=val_acc_mask)

                val_acc = val_acc_metric.result()
                val_b_soft_precision = val_b_soft_precision_metric.result()
                val_b_soft_recall = val_b_soft_recall_metric.result()
                val_i_soft_precision = val_i_soft_precision_metric.result()
                val_i_soft_recall = val_i_soft_recall_metric.result()
                #val_O_precision = val_O_precision_metric.result()
                #val_O_recall = val_O_recall_metric.result()
                val_acc_prev = val_acc
                val_b_soft_precision_prev = val_b_soft_precision
                val_b_soft_recall_prev = val_b_soft_recall
                val_i_soft_precision_prev = val_i_soft_precision
                val_i_soft_recall_prev = val_i_soft_recall
                #val_O_precision_prev = val_O_precision
                #val_O_recall_prev = val_O_recall
                val_acc_metric.reset_states()
                val_b_soft_precision_metric.reset_states()
                val_b_soft_recall_metric.reset_states()
                val_i_soft_precision_metric.reset_states()
                val_i_soft_recall_metric.reset_states()
                #val_O_precision_metric.reset_states()
                #val_O_recall_metric.reset_states()
                print('Validation acc: %s' % (float(val_acc),))
                print('val B-software precision over epoch: %s' % (float(val_b_soft_precision),))
                print('val B-software recall over epoch: %s' % (float(val_b_soft_recall),))
                print('val I-software precision over epoch: %s' % (float(val_i_soft_precision),))
                print('val I-software recall over epoch: %s' % (float(val_i_soft_recall),))
                #print('O precision over epoch: %s' % (float(val_O_precision),))
                #print('O recall over epoch: %s' % (float(val_O_recall),))
                tf.summary.scalar("Validation Accuracy", val_acc, step=global_epochs)
                tf.summary.scalar("val B-software Precision", val_b_soft_precision, step=global_epochs)
                tf.summary.scalar("val B-software Recall", val_b_soft_recall, step=global_epochs)
                tf.summary.scalar("val B-software Fscore", (2 * val_b_soft_precision * val_b_soft_recall) / (val_b_soft_precision + val_b_soft_recall), step=global_epochs)
                tf.summary.scalar("val I-software Precision", val_i_soft_precision, step=global_epochs)
                tf.summary.scalar("val I-software Recall", val_i_soft_recall, step=global_epochs)
                tf.summary.scalar("val I-software Fscore", (2 * val_i_soft_precision * val_i_soft_recall) / (val_i_soft_precision + val_i_soft_recall), step=global_epochs)
                #tf.summary.scalar("O Precision", val_O_precision, step=global_epochs)
                #tf.summary.scalar("O Recall", val_O_recall, step=global_epochs)
                validation_end_time = time.time()
                print("Validation took %s seconds" % (validation_end_time - validation_start_time))

            else: 
                validation_start_time = time.time()
                print('Start of custom eval for epoch %d' % (global_epochs,))
            
                cf = custom_evaluation(model, test_set)

                tf.summary.scalar("val B-software Precision", cf.loc[cf['entity']=='B-software', 'precision']['strict_B-software'], step=global_epochs)
                tf.summary.scalar("val B-software Recall", cf.loc[cf['entity']=='B-software', 'recall']['strict_B-software'], step=global_epochs)
                tf.summary.scalar("val B-software Fscore", cf.loc[cf['entity']=='B-software', 'f1']['strict_B-software'], step=global_epochs)

                tf.summary.scalar("val I-software Precision", cf.loc[cf['entity']=='I-software', 'precision']['strict_I-software'], step=global_epochs)
                tf.summary.scalar("val I-software Recall", cf.loc[cf['entity']=='I-software', 'recall']['strict_I-software'], step=global_epochs)
                tf.summary.scalar("val I-software Fscore", cf.loc[cf['entity']=='I-software', 'f1']['strict_I-software'], step=global_epochs)

                print(cf.drop(cf.columns.difference(['precision','recall', 'f1']), 1, inplace=False))
                cf = cf[cf['type']=='entity']

                tf.summary.scalar("val Ent-type Precision", cf.loc[cf['entity']=='all', 'precision']['ent_type'], step=global_epochs)
                tf.summary.scalar("val Ent-type Recall", cf.loc[cf['entity']=='all', 'recall']['ent_type'], step=global_epochs)
                tf.summary.scalar("val Ent-type FScore", cf.loc[cf['entity']=='all', 'f1']['ent_type'], step=global_epochs)

                tf.summary.scalar("val Exact Precision", cf.loc[cf['entity']=='all', 'precision']['exact'], step=global_epochs)
                tf.summary.scalar("val Exact Recall", cf.loc[cf['entity']=='all', 'recall']['exact'], step=global_epochs)
                tf.summary.scalar("val Exact FScore", cf.loc[cf['entity']=='all', 'f1']['exact'], step=global_epochs)

                tf.summary.scalar("val Strict Precision", cf.loc[cf['entity']=='all', 'precision']['strict'], step=global_epochs)
                tf.summary.scalar("val Strict Recall", cf.loc[cf['entity']=='all', 'recall']['strict'], step=global_epochs)
                tf.summary.scalar("val Strict FScore", cf.loc[cf['entity']=='all', 'f1']['strict'], step=global_epochs)

                validation_end_time = time.time()
                print("Custom eval took %s seconds" % (validation_end_time - validation_start_time))
            writer.flush()
            global_epochs += 1
    return global_epochs

# Training functions concerned with running all initializations and calling the training loop 
def run_training(args):
    # Creating the datasets
    if args.train_set == 'silver_train':
        train_set = create_dataset('../data/'+silver_standard_positive+'_data.txt', '../data/'+silver_standard_positive+'_labels.txt', shuffle=True, down_sample=False, take=args.take, skip=args.skip)
    elif args.train_set.startswith('silver_merged_'):
        set_number = args.train_set.split('_')[2]
        fold = args.train_set.split('_')[4]
        train_set = create_dataset('../data/merged_silver_standard_data_ep{}_{}.txt'.format(set_number, fold), '../data/merged_silver_standard_labels_ep{}_{}.txt'.format(set_number, fold), shuffle=True, down_sample=False, take=args.take, skip=args.skip)
#    elif args.train_set == 'silver_merged_train_2':
#        train_set = create_dataset('../data/'+merged_silver_standard_2+'_data.txt', '../data/'+merged_silver_standard_2+'_labels.txt', shuffle=True, down_sample=False)
    elif args.train_set == 'gold_train_with_pos':
        train_set = create_dataset('../data/sosci_train_with_pos_data.txt', '../data/sosci_train_with_pos_labels.txt', shuffle=True, down_sample=False, take=args.take, skip=args.skip)
    elif args.train_set == 'gold_train_no_pos':
        train_set = create_dataset('../data/sosci_train_no_pos_data.txt', '../data/sosci_train_no_pos_labels.txt', shuffle=True, down_sample=False, take=args.take, skip=args.skip)
    elif args.train_set == 'gold_devel':
        train_set = create_dataset(['../data/'+gold_standard_train+'_data.txt', '../data/'+gold_standard_devel+'_data.txt'], ['../data/'+gold_standard_train+'_labels.txt', '../data/'+gold_standard_devel+'_labels.txt'], shuffle=True, down_sample=False, take=args.take, skip=args.skip)
    elif args.train_set.startswith('silver_train_'):
        extension = args.train_set.split('_')[-1]
        train_set = create_dataset('../data/merged_silver_standard_data_ep{}.txt'.format(extension), '../data/merged_silver_standard_labels_ep{}.txt'.format(extension), shuffle=True, down_sample=False, take=args.take, skip=args.skip)
    elif args.train_set == 'duck':
        train_set = create_dataset('../data/'+args.train_set+'_data.txt', '../data/'+args.train_set+'_labels.txt', shuffle=True, down_sample=False, take=args.take, skip=args.skip)
    elif args.train_set.startswith('silver_opt_'):
        extension = args.train_set.split('_')[-1]
        train_set = create_dataset('../data/merged_silver_opt_train_data_ep{}.txt'.format(extension), '../data/merged_silver_opt_train_labels_ep{}.txt'.format(extension), shuffle=True, down_sample=False, take=args.take, skip=args.skip)
    else:
        raise(RuntimeError("Unknown data selection was passed as train set."))

    if args.test_set == 'gold_devel':
        test_set = create_dataset('../data/'+gold_standard_devel+'_data.txt', '../data/'+gold_standard_devel+'_labels.txt', shuffle=False, down_sample=False)
    elif args.test_set == 'gold_test':
        test_set = create_dataset('../data/'+gold_standard_test+'_data.txt', '../data/'+gold_standard_test+'_labels.txt', shuffle=False, down_sample=False)
    elif args.test_set == 'silver_opt':
        test_set = create_dataset('../data/merged_silver_opt_test_data.txt', '../data/merged_silver_opt_test_labels.txt', shuffle=False, down_sample=False)
    else:
        raise(RuntimeError("Unknown data selection was passed as test set."))

    # Creating the model (with the necessary pretrained word embedding)
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
    print('Found {} of {} words in word embedding ({} not found).'.format(found, vocab_size, not_found))
    word_embedding = None
    embedding_layer = Embedding(vocab_size,
                                EMBEDDING_SIZE,
                                embeddings_initializer=Constant(embedding),
                                trainable=False)

    initializer = GlorotUniform()
    model = bi_LSTM_seq_tagger(initializer=initializer,
                               embedding_layer=embedding_layer,
                               drop_rate=float(args.dropout_rate), 
                               char_num=character_size, 
                               char_embedding_size=CHAR_EMBEDDING_SIZE, 
                               char_lstm_size=CHAR_LSTM_SIZE,
                               lstm_size=int(args.lstm_size),
                               label_size=label_size)

    # Loading model weights if we start training from a checkpoint
    if args.checkpoint:
        print("Resuming training from checkpoint {}.".format(args.checkpoint))
        model.load_weights(join(CHECKPOINT_DIR, args.checkpoint, 'model'))
        print("Loaded weights.")
    else: 
        config = model.get_config()
        with open(join(CHECKPOINT_DIR, args.save_name + '_config.json'), 'w') as outfile:
            json.dump(config, outfile, indent=4)

    # Calling the training loop
    if not args.global_epoch:
        global_epochs = 0
    else:
        global_epochs = int(args.global_epoch)
    print("Starting traing from epoch {} for {} epochs on {} set and validating on {} set with learning rate {}.".format(global_epochs, args.epochs, args.train_set, args.test_set, args.learning_rate))
    global_epoch_count = train_test_func(model, train_set, test_set, int(args.epochs), float(args.learning_rate), float(args.learning_decay), global_epochs, args.save_name, args.log_name, train_test=args.train, train_logging=bool(int(args.log_training)), val_freq=1, sample_weighting=args.sample_weights, custom_eval=args.custom_eval)
    return global_epoch_count, model, train_set, test_set

def custom_evaluation(model, test_set):
    y_true = []
    y_pred = []
    for x, y in test_set:
        _, _, sequence_length, tag_seq = model(x, training=False)
        y_true_sents = tf.split(y, y.shape[0])
        y_pred_sents = tf.split(tag_seq, y.shape[0])
        sentence_lengths = tf.split(sequence_length, y.shape[0])
        for sent_true, sent_pred, length in zip(y_true_sents, y_pred_sents, sentence_lengths):
            l = tf.squeeze(length)
            y_true.extend(label_encoder.decode(tf.squeeze(sent_true)[0:l].numpy()).split())
            y_pred.extend(label_encoder.decode(tf.squeeze(sent_pred)[0:l].numpy()).split())

    df = evaluate_flat(y_true, y_pred)
    #df = get_main_scores(df)
    #print(df)
    return(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Train and test a Bi-LSTM-CRF model on silverstandard or gold standard.")
    
    parser.add_argument("--learning-rate", required=True, help="Learning rate to be used across all epochs")
    parser.add_argument("--learning-decay", required=True, help="Learning rate decay per epoch.")
    parser.add_argument("--dropout-rate", required=True, help="Set the dropout-rate for training")
    parser.add_argument("--epochs", required=True, help="Number of training epochs")
    parser.add_argument("--train-set", required=True, help="Training set to be used: silver_train or gold_train")
    parser.add_argument("--test-set", required=True, help="Testset to be used, gold_train, gold_devel or gold_trest")
    parser.add_argument("--checkpoint", required=False, help="Model weights to continue training from")
    parser.add_argument("--global_epoch", required=False, help="When continue training provide global epoch so tensorboard logging does not get confused")
    parser.add_argument("--save-name", required=True, help="Output name for the model")
    parser.add_argument("--log-name", required=True, help="Logging name")
    parser.add_argument("--train", required=True, help="Choose test or train mode")
    parser.add_argument("--log-training", required=True, help="Activate full logging on training")
    parser.add_argument("--custom-eval", required=True, help="Determines if a custom evaluation is performed on the final output model.")
    parser.add_argument("--take", required=False, help="Take only a part of the dataset, expect integer value of samples to take.")
    parser.add_argument("--skip", required=False, help="Skip part of the dataset, expect integer number of samples to skip.")
    parser.add_argument("--sample-weights", required=False, help="Put a stronger weight on positive samples. Expects a floating point difference from 1.0. Positive samples will be weighted by 1+diff. Negative with 1-diff.")
    parser.add_argument("--lstm-size", required=True, help="Set size of LSTM")
    args = parser.parse_args()

    # Setting up the save directory
    if not exists:
        mkdir(join(CHECKPOINT_DIR, args.save_name))
    
    global_epoch_count, model, _, test_set = run_training(args)

    sys.exit(global_epoch_count)
