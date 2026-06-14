from pathlib import Path
import json
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM


class AxionModelSplitter:
    def __init__(self, model_name: str, output_dir: str):
        self.model_name = model_name
        self.output_dir = Path(output_dir)

    def split(self):
        print(f"Loading model: {self.model_name}")

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="cpu"
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        layers_dir = self.output_dir / "layers"
        layers_dir.mkdir(exist_ok=True)

        config_path = self.output_dir / "config.json"

        with open(config_path, "w") as f:
            json.dump(model.config.to_dict(), f, indent=2)

        print("Saving embedding layer...")

        embed_tensors = {
            "model.embed_tokens.weight": model.model.embed_tokens.weight.detach().cpu()
        }

        save_file(embed_tensors, str(layers_dir / "embedding.safetensors"))

        print("Saving transformer layers...")

        for idx, layer in enumerate(model.model.layers):
            state_dict = {
                k: v.detach().cpu()
                for k, v in layer.state_dict().items()
            }

            layer_path = layers_dir / f"layer_{idx}.safetensors"

            save_file(state_dict, str(layer_path))

            print(f"Saved layer {idx}")

        print("Saving final norm...")

        norm_tensors = {
            "model.norm.weight": model.model.norm.weight.detach().cpu()
        }

        save_file(norm_tensors, str(layers_dir / "norm.safetensors"))

        print("Saving lm_head...")

        lm_head_tensors = {
            "lm_head.weight": model.lm_head.weight.detach().cpu()
        }

        save_file(lm_head_tensors, str(layers_dir / "lm_head.safetensors"))

        print("Axion split complete")


if __name__ == "__main__":
    splitter = AxionModelSplitter(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        output_dir="models/cache/qwen2.5-3b"
    )

    splitter.split()