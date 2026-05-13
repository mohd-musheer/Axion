
import axion_cpp

from inference.python.tokenizer import (
    AxionTokenizer
)


class AxionGenerator:

    def __init__(
        self,
        model_path,
        tokenizer_name="gpt2"
    ):

        self.tokenizer = AxionTokenizer(
            tokenizer_name
        )

        self.loader = axion_cpp.MMapLoader()

        self.loader.load_file(
            model_path
        )

        # GPT2 embeddings

        self.embedding_matrix = (
            self.loader.load_tensor_data(
                "wte.weight"
            )
        )

    def forward(
        self,
        token_ids
    ):

        hidden = axion_cpp.embedding_lookup(
            self.embedding_matrix,
            token_ids
        )

        logits = axion_cpp.compute_logits(
            hidden,
            self.embedding_matrix
        )

        next_token = axion_cpp.argmax(
            logits
        )

        return next_token

    def generate(
        self,
        prompt,
        max_new_tokens=10
    ):

        token_ids = self.tokenizer.encode(
            prompt
        )

        print("\nINITIAL TOKENS:")

        print(token_ids)

        for step in range(max_new_tokens):

            next_token = self.forward(
                token_ids
            )

            token_ids.append(
                next_token
            )

            decoded = self.tokenizer.decode(
                [next_token]
            )

            print(
                f"\nSTEP {step+1}"
            )

            print(
                "TOKEN:",
                next_token
            )

            print(
                "TEXT:",
                repr(decoded)
            )

        return self.tokenizer.decode(
            token_ids
        )