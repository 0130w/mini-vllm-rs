import torch
from transformers import AutoConfig, Qwen3_5ForConditionalGeneration


def main():
    model_dir = "../models/qwen-3.5-0.8B/"
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    with torch.device("meta"):
        # see config.json
        model = Qwen3_5ForConditionalGeneration(config)

    with open(f"{model_dir}/model_structure.txt", "w") as f:
        f.write(str(model))


if __name__ == "__main__":
    main()
