# Configuration

The demo can be launched either with CLI flags or with YAML files in `configs/`.

`configs/mbpp_las_distilbert.yaml` runs the paper-aligned MBPP LAS workflow:

- executor backend: Doubao / Ark (`--llm_api doubao`)
- LAS scheduler backend: Doubao / Ark (`--las_llm_api doubao`)
- gate: DistilBERT checkpoint plus local agreement, spec adherence, and history features
- output: one CSV per task plus `summary.csv` and `run_config.json`

The tracked gate file is a Git LFS pointer. Put the full trained weights at:

```text
artifacts/gate/best_model_state.pt
```

or pass another path with `--gate_checkpoint_path`.

Secrets belong in `.env`, using `.env.example` as the template. The config runner loads `.env` automatically when it exists.

## Local models

Install with `pip install -e ".[local]"`. Use `--llm_api local --model <name>` for either runner. With LAS, use `--las_llm_api local --las_model <name>` to select an independent router model. Use `--max_tokens` and `--las_max_tokens` to set the output caps. The router inherits the worker backend, model, and cap when the separate flags are omitted. The local smoke YAML files select distinct worker and router profiles.

| Tier | Model | Default API base | Environment prefix |
| --- | --- | --- | --- |
| 0 | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` | `http://192.168.0.211:80/v1` | `NEMOTRON_NANO` |
| 0 | `openai/gpt-oss-20b` | `http://192.168.0.215:80/v1` | `GPT_OSS_20B` |
| 0 | `Qwen/Qwen3.5-35B-A3B-FP8` | `http://192.168.0.217:80/v1` | `QWEN_35B` |
| 1 | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8` | `http://192.168.0.209:80/v1` | `NEMOTRON_SUPER` |
| 1 | `openai/gpt-oss-120b` | `http://192.168.0.210:80/v1` | `GPT_OSS_120B` |
| 1 | `Qwen/Qwen3.5-122B-A10B-FP8` | `http://192.168.0.205:80/v1` | `QWEN_122B` |

Override each host through `<PREFIX>_BASE_URL` and set `<PREFIX>_API_KEY` if the endpoint uses bearer authentication. Empty keys omit the authorization header. Local requests go directly to the configured endpoints. Run from a host that can reach them.

For a checkpoint-free gate control, use `--untrained_gate --gate_seed 0` and omit `gate_checkpoint_path` from the configuration and command. This retains pretrained DistilBERT base weights and initializes a fresh classification head.
