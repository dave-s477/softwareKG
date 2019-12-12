from os import listdir
from os.path import join, basename, splitext
import re
from collections import namedtuple
import sys
import nltk
import json
import argparse
from shutil import copyfile

def transform_positive_samples(input_folder, output_folder, positive_samples_file, remove):
    with open(positive_samples_file, 'r') as file:
        positive_samples = json.load(file)
    for document in positive_samples:
        with open(join(input_folder, document + '.txt'), 'r') as in_txt, open(join(input_folder, document + '.ann'), 'r') as in_ann, open(join(input_folder, document + '.src'), 'r') as in_src, \
                open(join(output_folder, document + '.txt'), 'w') as out_txt, open(join(output_folder, document + '.ann'), 'w') as out_ann, open(join(output_folder, document + '.src'), 'w') as out_src:
            annotation_line_idx, _, _, _ = positive_samples[document].split(':', maxsplit=3)
            annotations = []
            for line in in_ann:
                if "AnnotatorNotes" in line:
                    continue
                anno_id, add_info, plain_text = line.split('\t')
                anno_label, off_beg, off_end = add_info.split()
                annotations.append({'id': anno_id, 'anno_label': anno_label, 'off_beg': int(off_beg), 'off_end': int(off_end), 'plain_text': plain_text})
            offset = 0
            for idx, line in enumerate(in_txt):
                if idx != int(annotation_line_idx):
                    out_txt.write(line)
                    offset += len(line)
                else:
                    # Need to skip this line... 
                    skip_onset = offset
                    skip_end = offset + len(line)
                    skipped_chars = len(line)
            for idx, line in enumerate(in_src):
                if idx != int(annotation_line_idx):
                    out_src.write(line)
            for annotation in annotations:
                if annotation['off_beg'] < skip_onset and annotation['off_end'] < skip_end:
                    out_ann.write('{}\t{} {} {}\t{}'.format(annotation['id'], annotation['anno_label'], annotation['off_beg'], annotation['off_end'], annotation['plain_text']))
                elif annotation['off_beg'] > skip_onset and annotation['off_end'] > skip_end:
                    #print("File {}".format(document))
                    out_ann.write('{}\t{} {} {}\t{}'.format(annotation['id'], annotation['anno_label'], annotation['off_beg']-skipped_chars, annotation['off_end']-skipped_chars, annotation['plain_text']))
    all_files = [x.split('.txt')[0] for x in listdir(input_folder) if x.endswith('.txt')]
    files_without_positive_samples = set(all_files) - set(list(positive_samples.keys()))
    for document in files_without_positive_samples:
        copyfile(join(input_folder, document+'.txt'), join(output_folder, document+'.txt'))
        copyfile(join(input_folder, document+'.ann'), join(output_folder, document+'.ann'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Converts Brat Annotation Format to Bio annotation format")

    parser.add_argument("--input-folder", required=True, help="Folder that contains the brat annotated files (txt and ann are assumed to have the same basename)")
    parser.add_argument("--output-folder", required=True, help="Filename of the output file for BIO annotation format")
    parser.add_argument("--positive-samples", default='positive_samples_overview.json', help="Filename of positive samples to be removed that were introduced for annotation")
    parser.add_argument("--remove", action='store_true', help="Remove positive samples, if not set they will be merged.")

    args = parser.parse_args()

    print(args)

    transform_positive_samples(input_folder = args.input_folder,
                                output_folder = args.output_folder,
                                positive_samples_file = args.positive_samples,
                                remove = args.remove)