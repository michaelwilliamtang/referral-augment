import torch

from transformers import AutoTokenizer, AutoModel

class HuggingFaceEncoder:
    def __init__(self, doc_model_string, doc_tokenizer_string=None,
                 query_model_string=None, query_tokenizer_string=None):
        '''
        doc_model_string: HuggingFace model string for doc encoder
        doc_tokenizer_string: HuggingFace tokenizer string for doc encoder,
            or None if same as doc model string
        query_model_string: HuggingFace model string for query encoder,
            or None if same as doc model string
        query_tokenizer_string: HuggingFace tokenizer string for query encoder,
            or None if same as query model string
        '''
        # if doc and query model are the same
        if query_model_string is None:
            query_model_string = doc_model_string
        
        # if tokenizers have the same name as models
        if doc_tokenizer_string is None:
            doc_tokenizer_string = doc_model_string
        if query_tokenizer_string is None:
            query_tokenizer_string = query_model_string
        
        self.doc_model = AutoModel.from_pretrained(doc_model_string)
        self.doc_tokenizer = AutoTokenizer.from_pretrained(doc_tokenizer_string)
        self.query_model = AutoModel.from_pretrained(query_model_string)
        self.query_tokenizer = AutoTokenizer.from_pretrained(query_tokenizer_string)
        
        self.doc_model.eval()
        self.query_model.eval()

    def _encode(self, x, model, tokenizer):
        # preprocess the input
        inputs = tokenizer(x, padding=True, truncation=True, return_tensors="pt",
                           return_token_type_ids=False, max_length=512)
        with torch.no_grad():
            output = model(**inputs)

        # take the embedding of the last token
        assert output[-1].shape[1] == 768
        return output[-1].detach().numpy()
    
    def encode(self, x, is_query=False):
        '''
        is_query: whether x is query text (else, is document text)
        '''
        if is_query: # use query model
            return self._encode(x, self.query_model, self.query_tokenizer)
        else: # use doc model
            return self._encode(x, self.doc_model, self.doc_tokenizer)
