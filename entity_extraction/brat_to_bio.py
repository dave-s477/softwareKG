from os import listdir
from os.path import join, basename, splitext
import re
from collections import namedtuple
import sys
import nltk
import json
import argparse

output = {}
positive_samples = {}

# NERsuite tokenization: any alnum sequence is preserved as a single
# token, while any non-alnum character is separated into a
# single-character token. TODO: non-ASCII alnum.
TOKENIZATION_REGEX = re.compile(r'((?:10.1371.journal.[a-z]+.[a-z0-9\.]+)|\[[0-9\-,\u2013\?]+\]|[0-9]*\.[0-9]+|[a-zA-Z]+[0-9a-zA-Z]+|[a-zA-Z]+|[^0-9a-zA-Z])')

# tokenization based on whitespaces
def tokenize(line):
    return [t for t in TOKENIZATION_REGEX.split(line) if t]

def nltk_tokenize(line):
    return nltk.tokenize.word_tokenize(line)

TEXTBOUND_LINE_RE = re.compile(r'^T\d+\t')

Textbound = namedtuple('Textbound', 'start end type text')

def get_label(label, use_uncertain):
    label_parts = label.split('_')
    if len(label_parts) <= 1:
        return label
    elif use_uncertain:
        return label_parts[0]
    elif len(label_parts) == 2 and label_parts[1].startswith('certain'):
        return label_parts[0]
    else:
        return 'O'

def parse_textbounds(f):
    """Parse textbound annotations in input, returning a list of Textbound."""

    textbounds = []

    for l in f:
        l = l.rstrip('\n')

        if not TEXTBOUND_LINE_RE.search(l):
            continue

        _ , type_offsets, text = l.split('\t')
        atype, start, end = type_offsets.split()
        start, end = int(start), int(end)

        textbounds.append(Textbound(start, end, atype, text))

    return textbounds

def get_annotations(annfn):
    with open(annfn, 'rU') as f:
        textbounds = parse_textbounds(f)

    return textbounds

def relabel(document, annotations, use_uncertain):

    # TODO: this could be done more neatly/efficiently
    offset_label = {}
    output = []

    pos_software_counter = 0

    ti = 0
    for tb in annotations:
        tb = Textbound(tb.start, tb.end, tb.type + str(ti), tb.text)
        ti += 1
        for i in range(tb.start, tb.end):
            if i in offset_label:
                print("Warning: overlapping annotations", file=sys.stderr)
            offset_label[i] = tb

    prev_label = None
    for ii, lines in enumerate(document):
        for i, l in enumerate(lines):
            if not l:
                prev_label = None
                continue
            tag, start, end, token = l

            label = None
            for o in range(start, end):
                if o in offset_label:
                    if o != start:
                        print('Warning: annotation-token boundary mismatch: "%s" --- "%s"' % (
                            token, offset_label[o].text), file=sys.stderr)
                    if o + len(token) != end:
                        print("SEcond errror")
                    label = offset_label[o].type
                    break


            if label is not None:
                l = get_label(label, use_uncertain)
                if l == 'O':
                    tag = l
                elif label == prev_label:
                    tag = 'I-' + l
                else:
                    tag = 'B-' + l
                    pos_software_counter += 1
            prev_label = label

            lines[i] = [tag, start, end, token]
        output.append(lines)
        assert(len(output) == ii + 1)

    return output, pos_software_counter

def create_bio_from_brat(input_folder, output_bio, use_uncertain, files_shuffled, positive_samples_file, remove_duplicates, gather_pos_samples):
    if gather_pos_samples:
        print("""Positive Samples are written as output. 
        WARNING: the produced output does also contain the positive samples! 
        So right now this script needs to be run twice with different settings!
        First run with --write-pos flag, then without it.""")
        file_handle = open('data/positive_samples_bio.txt', 'w')
        file_handle.write('-DOCSTART- :positive_samples\n')

    total_software_annotations = 0

    annotation_filenames = [filename for filename in listdir(input_folder) if filename.endswith(".ann")]

    positive_samples = {}
    if positive_samples_file is not None:
        with open(positive_samples_file) as file:
            positive_samples = json.load(file)

    output_documents = {}
    all_src = []

    for annotation_filename in annotation_filenames:
        filename,_ = splitext(annotation_filename)
        text_filename = filename + ".txt"
        
        src_filename = filename + ".src"

        src = []
        if files_shuffled or remove_duplicates:
            with open(join(input_folder,src_filename)) as src_file:
                src = src_file.read().splitlines()
                all_src.extend(src)

        if filename in positive_samples:
            positive_samples_idx = int(positive_samples[filename].split(':')[0])
            positive_sample_word = positive_samples[filename].split(':')[3].split()[0]
        else:
            positive_samples_idx = -1

        with open(join(input_folder,text_filename)) as text_file:
    
            document = []
            line_idx = -1
            offset = 0
            for line in text_file:
                line_idx += 1
                
                if line_idx == positive_samples_idx and not gather_pos_samples:
                    # skip this sentence, as it was randomly introduced
                    offset += len(line)
                    assert(line.startswith(positive_sample_word))
                    continue

                # in the preprocess we added some empty lines by 'chance'
                if line == '\n':
                    if files_shuffled and not src[line_idx] == '':
                        print("Source file and text file do not match in line :", line_idx, text_filename, src_filename)
                    else:
                        offset += 1
                        continue

                if remove_duplicates:
                    if all_src.count(src[line_idx]) > 1:
                        offset += len(line)
                        continue

                lines = []
                tokens = tokenize(line)
                for t in tokens:
                    if not t.isspace():
                        lines.append(['O', offset, offset + len(t), t])
                    offset += len(t)
                document.append(lines)

            #now we could start the relabelling according to the ann file    
        document, pos_software_counter = relabel(document, get_annotations(join(input_folder,annotation_filename)), use_uncertain)
        total_software_annotations += pos_software_counter

        num_empty_lines = 0
        for i, lines in enumerate(document):
            if i == positive_samples_idx:
                if gather_pos_samples:
                    for token in lines:
                        file_handle.write(token[3] + '\t' + token[0]+'\n')
                    file_handle.write('\n')
                num_empty_lines += 1
            if files_shuffled:
                # files are either shuffled, which mean a corresponding file has the source
                if src[i + num_empty_lines] == '':
                    num_empty_lines += 1
                source_document, line = src[i + num_empty_lines].split(':')
            else:
                # or not shuffled, which mean, the filename points to the document
                source_document = filename
                line = i

            if source_document in output_documents.keys():
                output_documents[source_document][int(line)] = lines
            else:
                tmp = {int(line) : lines}
                output_documents[source_document] = tmp

    
    # write documents to file
    with open(output_bio, 'w') as o:
        documents = sorted(output_documents.keys())
        for document in documents:
            lines = output_documents[document]
            o.write('-DOCSTART- :'+ document +'\n')
            line_numbers = sorted(lines.keys())
            for n in line_numbers:
                for token in lines[n]:
                    o.write(token[3] + '\t' + token[0]+'\n')
                o.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Converts Brat Annotation Format to Bio annotation format")

    parser.add_argument("--input-folder", required=True, help="Folder that contains the brat annotated files (txt and ann are assumed to have the same basename)")
    parser.add_argument("--output-file", required=True, help="Filename of the output file for BIO annotation format")
    parser.add_argument("--use-uncertain", action='store_true', help="Includes also annotations that were marked as 'uncertain'")
    parser.add_argument("--positive-samples", default=None, help="Filename of positive samples to be removed that were introduced for annotation")
    parser.add_argument("--files-shuffled", action='store_true', help="Signals whether the files annotation files are shuffled or not")
    parser.add_argument("--rm-duplicates", action='store_true', help="remove duplicates when not shuffled")
    parser.add_argument("--write-pos", action='store_true')

    args = parser.parse_args()

    print(args)

    create_bio_from_brat(input_folder = args.input_folder,
                            output_bio = args.output_file,
                            use_uncertain = args.use_uncertain,
                            files_shuffled = args.files_shuffled,
                            positive_samples_file = args.positive_samples,
                            remove_duplicates = args.rm_duplicates,
                            gather_pos_samples = args.write_pos)