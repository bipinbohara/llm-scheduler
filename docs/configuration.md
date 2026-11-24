# Configuration

The demo can be launched either with CLI flags or with YAML files in `configs/`.

`configs/mbpp_las_distilbert.yaml` runs the paper-aligned MBPP LAS workflow:

- executor backend: Doubao / Ark (`--llm_api doubao`)
- LAS scheduler backend: Doubao / Ark (`--las_llm_api doubao`)
- gate: DistilBERT checkpoint plus local agreement, spec adherence, and history features
- output: one CSV per task plus `summary.csv` and `run_config.json`

The gate checkpoint is intentionally not committed. Put it at:

```text
artifacts/gate/best_model_state.pt
```

or pass another path with `--gate_checkpoint_path`.

Secrets belong in `.env`, using `.env.example` as the template. The config runner loads `.env` automatically when it exists.
