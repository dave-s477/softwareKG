from os.path import join
import argparse

import os
import pickle

from glob import glob
from shutil import copy

parser = argparse.ArgumentParser()
parser.add_argument("--db-type", required=True, help="Set type of database used by Snorkel.")
parser.add_argument("--data-loc", required=True, help="Postgres database name")
parser.add_argument("--brat-labels", required=True, help="Pass the location of the BRAT label folder.")
parser.add_argument("--database", required=True, help="Postgres database name")
args = parser.parse_args()

BASE_NAME = args.data_loc
print("Working on data split: {}".format(BASE_NAME))

if args.db_type == 'postgres':
    os.environ['SNORKELDB'] = 'postgres://snorkel:snorkel@localhost/' + args.database
elif args.db_type == 'sqlite':
    os.environ['SNORKELDB'] = 'sqlite:///' + args.database + '.db'

from snorkel import SnorkelSession
from snorkel.models import candidate_subclass

session = SnorkelSession()
software = candidate_subclass('software', ['software'])
set_mapping = {
    'devel': 0, 
    'new': 1
}

#if False:
from snorkel.parser import TextDocPreprocessor, CorpusParser
from snorkel.parser.spacy_parser import Spacy
from snorkel.models import Document, Sentence
from snorkel.candidates import Ngrams, CandidateExtractor
from snorkel.matchers import RegexMatchEach, RegexMatchSpan

ngrams_one = Ngrams(n_max=6)
software_matcher = RegexMatchEach(rgx=r'.*', longest_match_only=False)
cand_extractor = CandidateExtractor(software, [ngrams_one], [software_matcher])

#doc_preprocessor = TextDocPreprocessor('data/social_science_txt/') 
doc_preprocessor = TextDocPreprocessor('data/{}/'.format(BASE_NAME)) #('data/brat_files_just_txt/') 
corpus_parser = CorpusParser(parser=Spacy())
if args.db_type == 'sqlite':
    corpus_parser.apply(doc_preprocessor, parallelism=1)
else:
    corpus_parser.apply(doc_preprocessor, parallelism=64)
print("Documents:", session.query(Document).count())
print("Sentences:", session.query(Sentence).count())

docs = session.query(Document).order_by(Document.name).all()

dev_sents   = set()
new_sents = set()

for _, doc in enumerate(docs):
    if doc.name.startswith('sent'):
        for s in doc.sentences:
            dev_sents.add(s)
    else: 
        for s in doc.sentences:
            new_sents.add(s)
            
print("Working on " + str(len(dev_sents)))     
print("The set of new sentences contain {} sentences.".format(len(new_sents)))

for i, sents in enumerate([dev_sents, new_sents]):
    if args.db_type == 'sqlite':
        cand_extractor.apply(sents, split=i, parallelism=1)
    else:
        cand_extractor.apply(sents, split=i, parallelism=64)
    print("Number of candidates:", session.query(software).filter(software.split == i).count())
    
from util.brat_import import BratAnnotator

brat = BratAnnotator(session, software, encoding='utf-8') 
train_cands = session.query(software).filter(software.split!=set_mapping['new']).all()
brat.import_gold_labels(session, "data/{}/".format(args.brat_labels), train_cands)