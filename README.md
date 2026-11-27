# LLM-as-Scheduler: MBPP Demo

This repository contains the MBPP code-generation demo for **LLM-as-Scheduler: Agentic Workflow Dynamic Scheduling**.

The demo implements a fixed multi-agent MBPP workflow and the LAS variant that adds a gate plus an LLM scheduler for dynamic routing.

## Repository Layout

```text
las_mbpp/
  prompts.py                         Prompt builders for each agent stage
  config.py                          Repository paths and default artifact locations
  keys.py                            Lightweight .env loader
  workflows/
    mbpp_baseline.py                 Fixed full MBPP workflow
    mbpp_las_distilbert.py           Gate + LAS-routed MBPP workflow
  cli/
    run_from_config.py               YAML/JSON config launcher
configs/
  mbpp_las_distilbert.yaml           Paper-aligned LAS smoke config
  mbpp_baseline.yaml                 Fixed-workflow smoke config
docs/
  configuration.md                   Key, checkpoint, and config notes
  workflows.md                       Workflow summary
scripts/
  run_las_demo.sh
  run_baseline_demo.sh
tests/
```

## Setup

```bash
conda activate stabletools3
pip install -e ".[ark,gate,dev]"
cp .env.example .env
```

Fill `.env` with the API keys for the backend you use. The paper-aligned LAS config uses Doubao through Volcano Ark, so it needs `ARK_API_KEY`.

The checkout contains a Git LFS pointer for the trained gate. Place the full checkpoint payload at:

```text
artifacts/gate/best_model_state.pt
```

or pass `--gate_checkpoint_path /path/to/best_model_state.pt`.

## Run

Small LAS smoke run:

```bash
python -m las_mbpp.cli.run_from_config configs/mbpp_las_distilbert.yaml
```

Fixed baseline smoke run:

```bash
python -m las_mbpp.cli.run_from_config configs/mbpp_baseline.yaml
```

Equivalent console scripts after `pip install -e .`:

```bash
las-mbpp-las --splits test --limit_per_split 1 --gate_checkpoint_path artifacts/gate/best_model_state.pt
las-mbpp-baseline --splits test --limit_per_split 1
```

Outputs are written under `results/<dataset>/<run_id>/`. Each completed task has a JSON record and stage CSV. Use `--resume <run_id>` with the original settings to continue a saved run and `--export_excel` to export the analysis workbook. Live journals, the progress summary, and the Perfetto trace use the default `experiment` stem.

See [architecture and command inventory](docs/architecture.md) for local inference, official dataset preparation, paired gate runs, and resume. See [result fields](docs/results.md) for metric definitions.

## Notes

- MBPP tasks are loaded from `google-research-datasets/mbpp` via Hugging Face `datasets`.
- Local evaluation executes generated Python code against MBPP tests. Run in an isolated environment.
- `run_multi_agent_mbpp.py` and `run_multi_agent_mbpp_las.py` remain as backward-compatible wrappers.
