from types import SimpleNamespace

class PreTrainedTokenizerBase:
    def __init__(self):
        self.vocab = {}

    def encode(self, text, add_special_tokens=False):
        tokens = text.lower().split()
        return [self._id(token) for token in tokens]

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=False, truncation=False):
        tokens = text.lower().split()
        ids = [self._id(t) for t in tokens]
        if return_offsets_mapping:
            offsets = []
            idx = 0
            lower = text.lower()
            for token in tokens:
                start = lower.find(token, idx)
                end = start + len(token)
                offsets.append((start, end))
                idx = end
            return SimpleNamespace(input_ids=ids, offset_mapping=offsets)
        return SimpleNamespace(input_ids=ids, offset_mapping=[(0,0)]*len(ids))

    def decode(self, token_ids, skip_special_tokens=True):
        inv_vocab = {v:k for k,v in self.vocab.items()}
        return " ".join(inv_vocab[i] for i in token_ids)

    def convert_tokens_to_string(self, tokens):
        return " ".join(tokens)

    def _id(self, token):
        if token not in self.vocab:
            self.vocab[token] = len(self.vocab)
        return self.vocab[token]

class AutoTokenizer:
    @staticmethod
    def from_pretrained(model_name):
        if "does-not-exist" in model_name:
            raise OSError(f"Model '{model_name}' not found")
        return PreTrainedTokenizerBase()
