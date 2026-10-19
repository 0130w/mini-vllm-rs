# 模型结构查看脚本

从本目录运行 `uv sync --locked` 和 `uv run dump_model.py`。

脚本读取本地 `../models/qwen-3.5-0.8B/config.json`，在 PyTorch 的 meta 设备上构建模型并输出结构，不加载模型权重。

模型来源、权重下载链接和项目运行说明见[项目 README](../README.md)。
