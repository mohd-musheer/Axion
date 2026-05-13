
class ModelAdapter:

    def embedding_name(self):
        raise NotImplementedError

    def q_proj_name(self, layer):
        raise NotImplementedError

    def k_proj_name(self, layer):
        raise NotImplementedError

    def v_proj_name(self, layer):
        raise NotImplementedError

    def o_proj_name(self, layer):
        raise NotImplementedError


class LlamaAdapter(ModelAdapter):

    def embedding_name(self):

        return "model.embed_tokens.weight"

    def q_proj_name(self, layer):

        return (
            f"model.layers.{layer}."
            f"self_attn.q_proj.weight"
        )

    def k_proj_name(self, layer):

        return (
            f"model.layers.{layer}."
            f"self_attn.k_proj.weight"
        )

    def v_proj_name(self, layer):

        return (
            f"model.layers.{layer}."
            f"self_attn.v_proj.weight"
        )

    def o_proj_name(self, layer):

        return (
            f"model.layers.{layer}."
            f"self_attn.o_proj.weight"
        )
