import nltk
import re

from os.path import join
from os import listdir

for f in listdir('TEXT'):
    print(f)
    with open(join('TEXT', f), 'r') as in_file, open(join('SENTS', f), 'w') as out_file:
        sentences = []
        for line in in_file:

            #flag = False
            #if 'e.g.' in line:
            #    flag = True
            #    print(line)

            if len(line.rstrip().lstrip()) > 0:
                line = re.sub(r'e\.g\.\s', r'e.g., ', line)
                line = re.sub(r'i\.e\.\s', r'i.e., ', line)
                sents = nltk.tokenize.sent_tokenize(line)
                #print(sents)
                #remove section headers which contain less that four words
                sents = [sent for sent in sents if not (sent.endswith(":") and len(sent.split(' ')) <= 4)]
                sentences.extend(sents)
                #print(sentences)
                #if flag:
                #    print(sents)
        for sent in sentences:
            out_file.writelines(sent + '\n')
        #break

