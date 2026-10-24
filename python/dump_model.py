from pathlib import Path

import torch
from torchview import draw_graph
from transformers import AutoConfig, Qwen3_5ForConditionalGeneration


def main():
    model_dir_path = Path(__file__).resolve().parents[1] / "models/qwen-3.5-0.8B"
    config = AutoConfig.from_pretrained(model_dir_path, local_files_only=True)
    with torch.device("meta"):
        # see config.json
        model = Qwen3_5ForConditionalGeneration(config)

    model_structure_path = model_dir_path / "model_structure.txt"

    if model_structure_path.exists() == False:
        model_structure_path.write_text(str(model), encoding="utf-8")

    example_inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        # Explicit masks for a four-token, unpadded prefill. This skips mask
        # preparation that reads tensor values, which meta tensors do not have.
        "attention_mask": {
            "full_attention": torch.triu(
                torch.full((1, 1, 4, 4), float("-inf")), diagonal=1
            ),
            "linear_attention": None,
        },
        "use_cache": False,
    }

    graph = draw_graph(
        model.model.language_model,
        input_data=example_inputs,
        expand_nested=True,
        depth=3,
        device="meta",
    )

    graph.visual_graph.render(str(model_dir_path / "model_graph"), format="svg")


if __name__ == "__main__":
    main()
