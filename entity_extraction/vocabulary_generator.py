import pickle
import argparse

from os import listdir, remove, mkdir
from os.path import join, exists

import tensorflow as tf 

from util.custom_token_encoder import CustomTokenTextEncoder

def main(train_locs, devel_loc, test_loc, padding, out_f, name):
    data = tf.data.TextLineDataset([x + 'data.txt' for x in train_locs] + [devel_loc + 'data.txt', test_loc +'data.txt'])
    labels = tf.data.TextLineDataset([x + 'labels.txt' for x in train_locs] + [devel_loc + 'labels.txt', test_loc +'labels.txt'])

    vocabulary_set = set()
    character_set = set()
    label_set = set()
    
    for text_tensor, char_tensor in tf.data.Dataset.zip((data, data)):
        some_tokens = tf.compat.as_text(text_tensor.numpy()).split()
        vocabulary_set.update(some_tokens)

        plain_text = tf.compat.as_text(char_tensor.numpy()).split()
        for token in plain_text:
            for char in token:
                character_set.update(char)
    
    for label_tensor in labels:
        lab = tf.compat.as_text(label_tensor.numpy()).split()
        label_set.update(lab)

    print("Working with {} words, {} characters and {} target labels.".format(len(vocabulary_set), len(character_set), len(label_set)))

    vocabulary_list = list(vocabulary_set)
    character_list = list(character_set)
    label_list = list(label_set)

    vocabulary_list.append('<PAD>')
    character_list.append('<PAD>')
    label_list.append('<PAD>')

    pickle.dump(vocabulary_list, open("{}/{}_word_voc.p".format(out_f, name), "wb" ))
    pickle.dump(character_list, open("{}/{}_char_voc.p".format(out_f, name), "wb" ))
    pickle.dump(label_list, open("{}/{}_label_voc.p".format(out_f, name), "wb" ))

    text_encoder = CustomTokenTextEncoder(vocabulary_set)
    character_encoder = CustomTokenTextEncoder(character_set)
    label_encoder = CustomTokenTextEncoder(label_set)

    text_encoder.save_to_file('{}/text_encoder'.format(out_f))
    character_encoder.save_to_file('{}/character_encoder'.format(out_f))
    label_encoder.save_to_file('{}/label_encoder'.format(out_f))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Generate a vocabulary.")

    parser.add_argument("--train-sets", required=True, nargs='+', help="Exact path to train-set to use.")
    parser.add_argument("--devel-set", required=True, help="Exact path to devel-set to use.")
    parser.add_argument("--test-set", required=True, help="Exact path to test-set to use.")
    parser.add_argument("--out-folder", required=True, help="Folder where the output will be written (will be generated if it does not exist).")
    parser.add_argument("--dataset-name", required=True, help="Base name for the dataset which will be used to name the files.")
    parser.add_argument("--use-padding", nargs='?', const=True, default=False, help="Add an explicit encoding for paddings.")

    args = parser.parse_args()

    if not exists(join('.', args.out_folder)):
        mkdir(join('.', args.out_folder))

    main(
        train_locs=args.train_sets,
        devel_loc=args.devel_set,
        test_loc=args.test_set,
        padding=args.use_padding,
        out_f=args.out_folder,
        name=args.dataset_name
    )