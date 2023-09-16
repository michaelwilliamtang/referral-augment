'''
Augment a given corpus with DocT5Query document augmentation.
'''

import os
import sys
import random
import argparse
import time, datetime
import json
from tqdm import tqdm
import numpy as np
import torch

from transformers import T5Tokenizer, T5ForConditionalGeneration

# reference: https://huggingface.co/doc2query/all-with_prefix-t5-base-v1
def get_docT5query(doc, prefix, num_queries, model, tokenizer):
    # generate
    text = prefix + ": " + doc
    input_ids = tokenizer.encode(text, max_length=384, truncation=True, return_tensors='pt')
    outputs = model.generate(
        input_ids=input_ids,
        max_length=64,
        do_sample=True,
        top_p=0.95,
        num_return_sequences=num_queries)
    
    # decode
    queries = []
    for i in range(len(outputs)):
        query = tokenizer.decode(outputs[i], skip_special_tokens=True)
        queries.append(query)

    return ' '.join(queries)

def get_docT5query_augmented_docs(docs, prefix, num_queries, model, tokenizer):
    docs_docT5query = []
    for doc in tqdm(docs):
        docs_docT5query.append(doc + ' [QUERIES] ' + get_docT5query(doc, prefix=prefix, num_queries=num_queries, model=model, tokenizer=tokenizer))
    return docs_docT5query

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', help='dataset name, currently supports acl and arxiv')
    parser.add_argument('-m', '--model_name', help='docT5query generation model name', default='doc2query/all-with_prefix-t5-base-v1')
    parser.add_argument('-p', '--prefix', help='docT5query generation input prefix, e.g. text2query or abstract2title', default='text2query')
    parser.add_argument('-n', '--num_queries', type=int, help='number of generated queries to augment each doc with', default=30)
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    args = parser.parse_args()

    if args.verbose:
        start_time = time.time()
    
    # read unaugmented docs
    with open('data/paper_retrieval/{}/corpus.json'.format(args.dataset), 'r') as f:
        docs = json.load(f)
    
    # build augmented docs
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    docs_docT5query = get_docT5query_augmented_docs(docs, args.prefix, args.num_queries, model, tokenizer)
    
    # save
    with open('data/paper_retrieval/{}/corpus_docT5query_{}.json'.format(args.dataset, args.prefix), 'w') as f:
        json.dump(docs_docT5query, f)
    if args.verbose:
        print('Took {} seconds'.format(time.time() - start_time))
