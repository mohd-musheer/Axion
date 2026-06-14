
from transformers import AutoTokenizer


class AxionTokenizer:

    def __init__(self, model_name):

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

    def encode(self, text):

        return self.tokenizer.encode(
            text,
            add_special_tokens=False
        )

    def decode(self, token_ids):

        return self.tokenizer.decode(
            token_ids
        )

    @property
    def vocab_size(self):

        return self.tokenizer.vocab_size
