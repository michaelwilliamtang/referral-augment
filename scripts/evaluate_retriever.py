'''
Evaluate retrieval gains due to RAR for a retriever on a given corpus + dataset.
'''

import argparse

from retrievers import BM25Retriever, DenseRetriever, AggregationType
from encoders import HuggingFaceEncoder
from utils import load_corpus, load_eval_dataset, evaluate_retriever

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--corpus', help='corpus name, e.g. arxiv or acl')
parser.add_argument('-d', '--dataset', help='evaluation corpus name, e.g. arxiv or acl -- by default, same as corpus', default=None)
parser.add_argument('-n', '--num_examples', help='num of evaluation examples -- by default, uses entire dataset', default=None)
parser.add_argument('-m', '--model', help='HuggingFace model string, if using dense retriever -- else, uses BM25 by default', default=None)
parser.add_argument('-t', '--tokenizer', help='HuggingFace tokenizer string, if using dense retriever -- by default, same as model string', default=None)
parser.add_argument('-a', '--aggregation', help='aggregation method, either concat, mean (dense only), or shortest_path -- by default, concat for BM25 and mean for dense models', default=None)
parser.add_argument('-r', '--num_referrals', type=int, help='(max) num referrals to sample per document, integer >= 1', default=30)
parser.add_argument('-w', '--doc_weight', type=int, help='doc_weight, amount to weight original document text compared to a single referral, integer >= 0', default=1)
parser.add_argument('-v', '--verbose', action='store_true')
args = parser.parse_args()

dense = args.model is not None
if args.dataset is None:
    args.dataset = args.corpus
if dense and args.tokenizer is None:
    args.tokenizer = args.model
if args.aggregation is None:
    if dense:
        args.aggregation = AggregationType.MEAN
    else:
        args.aggregation = AggregationType.CONCAT
else:
    if args.aggregation == 'concat':
        args.aggregation = AggregationType.CONCAT
    elif args.aggregation == 'mean':
        assert dense, 'Cannot use mean aggregation with non-dense retriever'
        args.aggregation = AggregationType.MEAN
    elif args.aggregation == 'shortest_path':
        args.aggregation = AggregationType.SHORTEST_PATH
    else:
        raise AssertionError('Invalid aggregation type')

if args.verbose:
    print(args)

docs, referrals = load_corpus(args.corpus)
queries, ground_truth = load_eval_dataset(args.dataset, num_examples=args.num_examples)

if dense:
    encoder = HuggingFaceEncoder(args.tokenizer, args.model)

    print('Without RAR...')
    retriever = DenseRetriever(encoder, docs, verbose=args.verbose)
    evaluate_retriever(retriever, queries, ground_truth)
    print()
    
    print('With RAR...')
    retriever = DenseRetriever(encoder, docs, referrals, aggregation=args.aggregation,
                               num_referrals=args.num_referrals, doc_weight=args.doc_weight,
                               verbose=args.verbose)
    evaluate_retriever(retriever, queries, ground_truth)
else:
    print('Without RAR...')
    retriever = BM25Retriever(docs, verbose=args.verbose)
    evaluate_retriever(retriever, queries, ground_truth)
    print()
    
    print('With RAR...')
    retriever = BM25Retriever(docs, referrals, aggregation=args.aggregation,
                              num_referrals=args.num_referrals, doc_weight=args.doc_weight,
                              verbose=args.verbose)
    evaluate_retriever(retriever, queries, ground_truth)
