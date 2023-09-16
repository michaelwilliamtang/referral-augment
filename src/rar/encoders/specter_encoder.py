from sentence_transformers import SentenceTransformer

class SpecterEncoder:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/allenai-specter')

    def encode(self, x, is_query=False):
        '''
        is_query: whether x is query text (else, is document text) -- unused
        '''
        return self.model.encode(x)
