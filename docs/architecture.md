# LAS code-generation runs

## Architecture

The fixed runner calls three debate workers, revises their candidates with peer outputs, selects a candidate with consensus, and runs reflection, verification, testing, and error correction. The corrected candidate is the normal final output. The LAS runner applies its existing weighted DistilBERT gate after intermediate stages. A passing gate consults the router, which can send the current candidate directly to the output, jump to testing and correction, or continue refinement. The early-exit output node forwards the prior candidate without another generation call.

The native YAML files retain their service, model, stage sequence, and generation settings. Local inference is opt-in through the existing provider settings. The six model profiles describe endpoint and tier. The explicit worker and router model settings determine which models run, and tier metadata does not introduce model selection.

The benchmark adapters load official data and supply a task, function signature, and evaluation contract to the existing workflows. For EvalPlus, the full official base suite supplies stage results. The additional suite scores the final candidate after routing and repair have finished. Additional inputs and reference outputs do not enter agent prompts. Native sanitized MBPP retains its existing prompt policy and test execution.

## Commands and verification

From the repository root in a supported Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[local,gate,evalplus,excel,dev]'
python -m pip check
python -m pytest -q
cp .env.example .env
```

Add the `ark` or `anthropic` extra for the existing cloud configurations. The packaging metadata in `pyproject.toml` defines the CLI commands and dependency bounds. The local runner reports the relevant extra when an optional dependency is missing.

The following inventory prepares the complete datasets and runs 1-item pilots. Remove `--limit_per_split` for full test runs. The checked-in smoke YAML files set a limit of one, so use the module CLI for full runs.

```bash
worker=openai/gpt-oss-20b
router=openai/gpt-oss-120b
checkpoint=/path/to/best_model_state.pt

for dataset in mbpp mbpp-plus humaneval humaneval-plus; do
  python -m las_mbpp.cli.prepare_data "$dataset"

  python -m las_mbpp.cli.run_baseline --dataset "$dataset" --splits test \
    --llm_api local --model "$worker" --max_tokens 4096 --limit_per_split 1

  python -m las_mbpp.cli.run_las --dataset "$dataset" --splits test \
    --llm_api local --model "$worker" --max_tokens 4096 \
    --las_llm_api local --las_model "$router" --las_max_tokens 4096 \
    --gate_checkpoint_path "$checkpoint" --limit_per_split 1

  python -m las_mbpp.cli.run_las --dataset "$dataset" --splits test \
    --llm_api local --model "$worker" --max_tokens 4096 \
    --las_llm_api local --las_model "$router" --las_max_tokens 4096 \
    --untrained_gate --gate_seed 0 --limit_per_split 1
done
```

Use the same `--splits`, `--task_ids`, and limit in each paired run. IDs are sorted numerically before the limit is applied. `run_config.json` saves the selected items and input fingerprints. `--task_ids` takes the numeric suffix of an official ID.

| Dataset flag | Source | Official splits and scoring |
| --- | --- | --- |
| `mbpp` | Google Research MBPP, Hugging Face `sanitized` config | `train`, `validation`, `test`, and `prompt`. Use the test split for reporting. Native assertion tests are evaluated without filtering. |
| `mbpp-plus` | EvalPlus MBPP+ v0.2.0 | Test only, 378 tasks. EvalPlus base and additional inputs, including EvalPlus special oracles and tolerances. |
| `humaneval` | Original OpenAI HumanEval through the EvalPlus data loader | Test only, 164 tasks. The official check function runs as a single suite for each task. |
| `humaneval-plus` | EvalPlus HumanEval+ v0.1.10 | Test only, 164 tasks. EvalPlus base and additional inputs, with EvalPlus scoring. |

No separate training or validation data is bundled for the test-only suites. Benchmark preparation uses the same official caches as the runners. If the Hugging Face dataset CDN is unavailable, prepare the pinned Google Research release with `python -m las_mbpp.cli.prepare_data mbpp --mbpp-json /path/to/sanitized-mbpp.json`, then export `MBPP_SANITIZED_JSON=/path/to/sanitized-mbpp.json` or set it in `.env`. The repository uses the original task-ID split boundaries and checks the release's SHA-256. Set `HF_HOME` and `XDG_CACHE_HOME` before downloading if the default caches are unwritable. Run candidate execution in an isolated environment. The runners' processes and EvalPlus guardrails do not provide an operating-system security boundary.

## Checkpoints and resume

A trained LAS checkpoint is the strict PyTorch state dict for the full two-label DistilBERT sequence classifier. Loading reads only weights and verifies tensor keys and shapes. Equal byte counts can reflect identical tensor shapes and dtypes with different values. Use the recorded SHA-256 for identity. The untrained control loads the configured base model and initializes a fresh classifier head using a fixed seed. The fixed pipeline has no classifier and uses no gate checkpoint.

Runs write to `results/<dataset>/<run-id>/` unless `--output_dir` supplies another root. The saved run configuration is written before execution. The experiment task journal is the durable completion record. Stage sidecars and the summary CSV refresh after each commit. See [experiment logs and results](results.md) for snapshot cadence, summaries, and timeline exports. Append `--resume <run-id>` to the original command to retain the same ID, verify settings and item fingerprints, rebuild missing exports, and skip completed tasks. Resume restores the LAS history from recorded gate observations. Tasks interrupted before saving are rerun. With concurrent tasks, restart can change the history observed by unfinished work because the native workflow updates history online as stages finish.

The workspace has the MBPP architecture reference and code-generation runners. It has no training pipeline, MATH-500, MMLU, GSM8K, MASBench, or HiddenBench integration, supplied analysis exports, or timing slides. Those requests need their corresponding experiment project, training objective, or source artifacts.

## Baseline and verification gaps

The designated base was `run-local` at `948199129619c9781d847534cbc89d498a30cbad`. All three existing unit tests passed before changes. Native demo startup reached the Ark and Anthropic credential checks. Regression checks cover both config entry points against a local OpenAI-compatible fixture, strict gate loading with a compatible tiny model, and interruption/resume. The trained release payload and the six private LAN services were unavailable in the verification environment, so the fixture verifies the integration without reporting benchmark accuracy for those services.

Each adapter also completed one official test item through the normal fixed-runner config launcher, using a loopback endpoint that replayed the reference implementation. MBPP passed 3/3 assertions. MBPP+ passed 3/3 base and 108/108 additional inputs. Original HumanEval passed its official check function. HumanEval+ passed 7/7 base and 999/999 additional inputs. Each run recorded 10 logical calls and 50 fixture tokens. These are integration checks with supplied answers.
