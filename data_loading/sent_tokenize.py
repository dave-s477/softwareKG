import nltk
import re

from os.path import join, exists
from os import listdir, makedirs

if not exists('./data/R_loading/SENTS'):
    makedirs('./data/R_loading/SENTS')

for f in listdir('./data/R_loading/TEXT'):
    print(f)
    with open(join('./data/R_loading/TEXT', f), 'r') as in_file, open(join('./data/R_loading/SENTS', f), 'w') as out_file:
        sentences = []
        for line in in_file:

            if len(line.rstrip().lstrip()) > 0:
                line = re.sub(r'e\.g\.\s', r'e.g., ', line)
                line = re.sub(r'i\.e\.\s', r'i.e., ', line)
                sents = nltk.tokenize.sent_tokenize(line)
                sents = [sent for sent in sents if not (sent.endswith(":") and len(sent.split(' ')) <= 4)]
                sentences.extend(sents)
        for sent in sentences:
            out_file.writelines(sent + '\n')