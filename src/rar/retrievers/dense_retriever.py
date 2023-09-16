import time
import random
import numpy as np
import pickle
from tqdm import tqdm

from .utils import AggregationType, SimilarityType

class DenseRetriever:
    def __init__(self, encoder, docs, referrals=None, num_referrals=30,
                 aggregation=AggregationType.CONCAT, doc_weight=1,
                 embeds_path=None, verbose=False):
        '''
        encoder: model with encode() method to encode strings into float arrays
        docs: list of document strings
        referrals: list of lists of referrals for each document, or None
        num_referrals: int >= 1, num referrals to use to augment each document
            representation (only used if referrals is not None)
        aggregation: Aggregation (only used if referrals is not None)
        doc_weight: int >= 0, weight to give to original document text, compared
            to referrals e.g. if 1, document text is weighted the same as a single referral
            if 0, document text is excluded and representation consists only of referrals
            (only applies if referrals is not None)
        embeds_path: filepath of embeddings pickle file to use directly instead of encoding
            at indexing time. For AggregationType.MEAN and AggregationType.CONCAT,
            embeddings must be np.ndarray with shape len(docs) x latent dim of encoder
        '''
        if verbose:
            print('Encoding corpus...')
            start_time = time.time()

        self.docs = []
        self.encoder = encoder
        self.aggregation = aggregation
        self.num_referrals = num_referrals
        self.embeds = []

        if embeds_path is not None:
            # use given embeddings directly
            if verbose:
                print('Using saved embeddings')
            with open(embeds_path, 'rb') as f:
                self.embeds = pickle.load(f)

        # encode to get embeddings, depending on type of referral
        if referrals is not None:
            assert isinstance(doc_weight, int) and doc_weight >= 0
            assert isinstance(num_referrals, int) and num_referrals >= 1
        if referrals is None:
            if verbose:
                print('Not using referral augmentation')
            if embeds_path is None:
                self.embeds = encoder.encode(docs, is_query=False)
            self.docs = np.array(docs)
        elif aggregation == AggregationType.SHORTEST_PATH:
            if verbose:
                print('Using shortest path referral augmentation, uniformly sampling {} referrals'
                      ' per document and encoding them separately'.format(num_referrals))
            for doc, referral_list in tqdm(zip(docs, referrals), disable=not verbose):
                # if less than num_referrals referrals, use all available
                referral_subset = random.sample(referral_list, min(num_referrals, len(referral_list)))
                keys = [' '.join([doc] * doc_weight + [referral]) for referral in referral_subset]
                if len(keys) == 0:
                    continue
                if embeds_path is None:
                    self.embeds.extend(encoder.encode(keys, is_query=False))
                self.docs.extend([doc] * len(keys))
            if embeds_path is None:
                self.embeds = np.array(self.embeds)
            self.docs = np.array(self.docs)
        elif aggregation == AggregationType.CONCAT:
            if verbose:
                print('Using concat referral augmentation, uniformly sampling {} referrals per document'
                      .format(num_referrals))
            if embeds_path is None:
                keys_list = []
                for doc, referral_list in tqdm(zip(docs, referrals), disable=not verbose):
                    # if less than num_referrals referrals, use all available
                    keys = [doc] * doc_weight + \
                        random.sample(referral_list, min(num_referrals, len(referral_list)))
                    keys_list.append(' '.join(keys))
                self.embeds = encoder.encode(keys_list, is_query=False)
            self.docs = np.array(docs)
        elif aggregation == AggregationType.MEAN:
            if verbose:
                print('Using mean referral augmentation, uniformly sampling {} referrals per document'
                      .format(num_referrals))
            if embeds_path is None:
                for doc, referral_list in tqdm(zip(docs, referrals), disable=not verbose):
                    # if less than num_referrals referrals, use all available
                    keys = [doc] * doc_weight + random.sample(referral_list,
                                                        min(num_referrals, len(referral_list)))
                    if len(keys) == 0:
                        keys = ['']
                        # encoders may not support empty lists
                        # (e.g. doc_weight = 0, no referrals for this doc)
                    self.embeds.append(encoder.encode(keys, is_query=False).mean(axis=0))
                self.embeds = np.array(self.embeds)
            self.docs = np.array(docs)

        if verbose:
            print('Took {} seconds'.format(time.time() - start_time))

    def retrieve(self, query, num_docs=10, similarity=SimilarityType.DOT):
        '''
        query: string
        num_docs: int, number of top documents to retrieve
        similarity: type of vector similarity (dot product or cosine)
        '''
        num_docs = min(num_docs, len(self.docs))

        # compute similarity
        encoded_query = self.encoder.encode(query, is_query=True)
        sims = self.embeds @ encoded_query.squeeze()
        if similarity == SimilarityType.COSINE:
            # normalize by norms
            norms = np.linalg.norm(self.embeds, axis=1) * np.linalg.norm(encoded_query)
            sims /= norms

        if self.aggregation == AggregationType.SHORTEST_PATH:
            # since we want num_docs unique documents, we retrieve more, then filter duplicates
            num_docs_before_filter = min(num_docs * self.num_referrals, len(self.docs))
            idxs = np.argpartition(sims, -num_docs_before_filter)[-num_docs_before_filter:]
            idxs = idxs[np.argsort(sims[idxs])[::-1]] # descending
            docs_before_filter = self.docs[idxs]
            return list(dict.fromkeys(docs_before_filter))[:num_docs]

        # get top num_docs -- note that argpartition is linear but the top k
        # are unsorted thus, we argpartition and then we sort post hoc
        # to get efficient + sorted top num_docs
        idxs = np.argpartition(sims, -num_docs)[-num_docs:]
        idxs = idxs[np.argsort(sims[idxs])[::-1]] # descending
        return self.docs[idxs]
