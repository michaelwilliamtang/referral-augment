import numpy as np
import json
import random

from .metrics import recall_at_k, mrr

def load_corpus(domain, corpus,
                corpus_fname='corpus', referrals_fname='referrals', return_referrals=True):
    '''
    Returns list of documents and list of lists of referrals
    '''
    with open('data/{}/{}/{}.json'.format(domain, corpus, corpus_fname), 'r') as f:
        docs = json.load(f)
    if not return_referrals:
        return docs
    with open('data/{}/{}/{}.json'.format(domain, corpus, referrals_fname), 'r') as f:
        referrals = json.load(f)
    return docs, referrals

def load_eval_dataset(domain, dataset, queries_fname='queries', ground_truth_fname='ground_truth',
                      num_examples=None):
    '''
    Returns lists of strings: queries, ground_truth
    num_examples: number of examples to randomly sample, or None (full dataset, randomly ordered)
    '''
    # load dataset -- we will use a subset of the ACL dataset as an example
    if num_examples is None:
        print('Loading full {} dataset on {} domain...'.format(dataset, domain))
    else:
        print('Loading {} samples from {} dataset on {} domain...'.format(num_examples, dataset, domain))

    with open('data/{}/{}/{}.json'.format(domain, dataset, queries_fname), 'r') as f:
        queries = json.load(f)
    with open('data/{}/{}/{}.json'.format(domain, dataset, ground_truth_fname), 'r') as f:
        ground_truth = json.load(f)

    if num_examples is None:
        num_examples = len(queries)
    random.seed(123)
    example_idxs = random.sample(range(len(queries)), num_examples)
    # ground_truth may be nested if multiple correct
    return np.array(queries)[example_idxs], [ground_truth[i] for i in example_idxs]

def evaluate_retriever(retriever, queries, ground_truth, include_stderr=True, multiple_correct=False):
    '''
    Prints evaluation metrics given a retriever and evaluation dataset
    include_stderr: whether to print stderr along with mean for each metric
    multiple_correct: whether each query has multiple ground truth docs,
        i.e. ground_truth is a list of lists instead of just a list (no MRR metric)
    '''
    predictions = [retriever.retrieve(query, num_docs=10) for query in queries]
    if include_stderr:
        if not multiple_correct:
            mean, std = mrr(ground_truth, predictions)
            print('MRR: {} +/- {}'.format(mean, std))
        mean, std = recall_at_k(ground_truth, predictions, k=1)
        print('Recall@1: {} +/- {}'.format(mean, std))
        mean, std = recall_at_k(ground_truth, predictions, k=10)
        print('Recall@10: {} +/- {}'.format(mean, std))
    else:
        if not multiple_correct:
            print('MRR: {}'.format(mrr(ground_truth, predictions, return_stderr=False)))
        print('Recall@1: {}'.format(recall_at_k(ground_truth, predictions, k=1, return_stderr=False)))
        print('Recall@10: {}'.format(recall_at_k(ground_truth, predictions, k=10, return_stderr=False)))
