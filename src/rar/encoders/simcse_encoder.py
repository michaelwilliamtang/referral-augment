# Requires installation of SimCSE here
# git clone https://github.com/princeton-nlp/SimCSE

from .SimCSE.simcse.tool import SimCSE

class SimCSEEncoder:
    def __init__(self):
        self.model = SimCSE('princeton-nlp/sup-simcse-roberta-large')

    def encode(self, x, is_query=False):
        '''
        is_query: whether x is query text (else, is document text) -- unused
        '''
        return self.model.encode(x).detach().numpy()
