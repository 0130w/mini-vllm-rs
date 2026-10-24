# mini-vllm

```bash
curl -fL --retry 3 -C - \
  'https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main/model.safetensors-00001-of-00001.safetensors' \
  -o models/qwen-3.5-0.8B/model.safetensors-00001-of-00001.safetensors
```

```bash
apt-get install -y --no-install-recommends libcurand-dev-13-0
```

```bash
cargo run
```

```bash
cd python
uv sync --locked
uv run dump_model.py
```