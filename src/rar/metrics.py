import numpy as np

def mrr(y, yhat, return_stderr=True):
    '''
    Computes mean reciprocal rank
    Note: ground truth y must be a single value
    '''
    mrrs = []
    for y_i, yhat_i in zip(y, yhat):
        if y_i not in yhat_i:
            mrrs.append(0)
            continue
        rank = 1 + np.where(np.array(yhat_i) == y_i)[0][0]
        mrrs.append(1.0/rank)
    if return_stderr:
        return np.mean(mrrs), np.std(mrrs) / np.sqrt(len(y))
    return np.mean(mrrs)

def _recall_at_k_single_correct(y, yhat, k, return_stderr):
    '''
    Computes recall@k
    Note: handles case when ground truth y is a single value
    '''
    recalls = []
    for y_i, yhat_i in zip(y, yhat):
        if y_i in yhat_i[:k]:
            recalls.append(1)
        else:
            recalls.append(0)
    if return_stderr:
        return np.mean(recalls), np.std(recalls) / np.sqrt(len(y))
    return np.mean(recalls)

def _recall_at_k_multiple_correct(y, yhat, k, return_stderr):
    '''
    Computes recall@k
    Note: handles case when ground truth y is a list (multiple correct / relevant docs)
    '''
    recalls = []
    for y_i, yhat_i in zip(y, yhat):
        # if *any* element of yhat_i matches *any* element of y_i, yield 1
        if len(set(y_i).intersection(set(yhat_i[:k]))) > 0:
            recalls.append(1)
        else:
            recalls.append(0)
    if return_stderr:
        return np.mean(recalls), np.std(recalls) / np.sqrt(len(y))
    return np.mean(recalls)

def recall_at_k(y, yhat, k=10, return_stderr=True):
    '''
    Computes recall@k
    Note: ground truth y can be a single value or a list (multiple correct / relevant docs)
    '''
    assert len(y) > 0, 'Cannot compute metrics on empty dataset'
    if isinstance(y[0], list):
        return _recall_at_k_multiple_correct(y, yhat, k, return_stderr)
    return _recall_at_k_single_correct(y, yhat, k, return_stderr)
