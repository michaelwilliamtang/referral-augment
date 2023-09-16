'''
Precompute embeddings for a given paper_retrieval corpus (e.g. acl or arxiv).
'''

import time
import argparse
import pickle
from sentence_transformers import SentenceTransformer

from retrievers import DenseRetriever, AggregationType
from utils import load_corpus

class SpecterEncoder:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/allenai-specter')

    # x: string to encode
    # returns: 1D np.array embedding
    def encode(self, x):
        return self.model.encode(x).numpy().detach()

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--corpus', help='corpus name, e.g. arxiv or acl')
parser.add_argument('-v', '--verbose', action='store_true')
args = parser.parse_args()
    
if args.verbose:
    start_time = time.time()
docs, referrals = load_corpus(args.corpus)
retriever = DenseRetriever(SpecterEncoder(), docs, referrals, aggregation=AggregationType.MEAN)
if args.verbose:
    print('Took {} seconds'.format(time.time() - start_time))

if args.verbose:
    start_time = time.time()
with open('data/paper_retrieval/{}/corpus_embeds.pickle'.format(args.corpus), 'wb') as f:
    pickle.dump(retriever.embeds, f)
if args.verbose:
    print('Took {} seconds'.format(time.time() - start_time))