# Experiment logs and results

Both the fixed and LAS launchers write a directory under `results/<dataset>/<UTC run ID>/`. Pass `--result_file /path/to/pilot.json` to select the directory and companion-file stem. Use one run per directory. `--export_excel` enables the workbook and is an alias for `--excel`. Install the `excel` extra first. `--export_every 25` changes the full snapshot interval. `--heartbeat_seconds 60` is the default, and zero disables the heartbeat.

```bash
python -m las_mbpp.cli.run_from_config configs/mbpp_local_baseline.yaml \
  --result_file results/mbpp/pilot/pilot.json --export_excel --export_every 25
```

| Default file | Contents and write cadence |
| --- | --- |
| `experiment.log` | Application DEBUG log with setup, effective settings, router decisions, task progress, worker inputs and outputs, grading diagnostics, generation-limit warnings, and final metrics. The sink opens before provider, gate, and dataset initialization. Console progress also goes to the legacy `run.log`. Capture stdout and stderr separately for dependency or subprocess output. |
| `experiment.events.jsonl` | Live task, node, call, round, route, answer reuse, and processing events, including span starts and finishes. Records are appended and synchronized as they occur. |
| `experiment.spans.jsonl` | Live completed spans, including failed and known-incomplete operations. Each completion is synchronized as it occurs. |
| `experiment.tasks.jsonl` | Authoritative task journal. Each committed row has the input, prediction, grade, route, node and call data, events, spans, and Python RNG state. |
| `experiment.summary.json` | Atomic progress and aggregate metrics at startup, each commit, and finalization. |
| `experiment.checkpoint.json` | Atomic resume cursor, signature, initial and last committed Python RNG state, and LAS snapshot identity. Updated on the same cadence as the summary. |
| `experiment.json` | Consolidated report with metadata, summary, and task records. Written at startup, every N commits, and finalization. |
| `experiment.trace.json` | Chrome Trace Event JSON for Perfetto. Built from committed task spans at full-snapshot intervals. |
| `experiment.xlsx` | Optional workbook with the same full-snapshot cadence. |
| `experiment.router.pth` | LAS classifier's effective weights, including untrained-head runs. Saved before any task. Resume checks the SHA-256 and strictly reloads the state dict. The fixed workflow has no router weights. |
| `experiment.heartbeat.jsonl` | Independent startup, timer, and handled-exit records. Reports committed and in-flight counts, active stage age, and last commit. Each heartbeat is synchronized. |
| `experiment.lock` | Kernel-backed single-writer guard. The file remains on disk after the lock is released. |
| `run_config.json`, per-task JSON/CSV, `summary.csv`, `summary.xlsx` | Existing interfaces. Task sidecars and the summary CSV refresh each commit. The optional legacy workbook follows the full-snapshot cadence. |

## Resume and trace recovery

Append `--resume RUN_ID` to the original command, including the same `--result_file` if one was used. The ID is printed when a run starts and recorded in `run_config.json`. Resume checks the ordered task fingerprints and effective settings, verifies LAS weights, restores Python RNG from the last durable task, and rebuilds progress from committed rows. Older runs are imported from their task JSON sidecars on their first resume. Unfinished tasks receive new attempt IDs. Their original events remain available in the live journals. Resume saves malformed final journal bytes to a `.tail-<id>.bak` file before removing them. Corruption inside a journal is an error.

A live trace can be recovered before or after resume. Event journals include starts for operations that were still open at process exit. The exporter treats those starts as incomplete and records the number of ignored tail bytes. It leaves the journal intact.

```bash
python -m las_mbpp.cli.export_trace \
  results/mbpp/pilot/pilot.events.jsonl results/mbpp/pilot/recovered.trace.json
```

## Join keys and timing

Use `run_id`, `split`, and `task_id` for the dataset item. `evaluation_index` is its zero-based position in the saved selection. `task_attempt` and `trace_id` identify an execution. `span_id` joins a span to its events, and `parent_span_id` identifies its immediate parent. `call_id` links model, node, route, and candidate grade records. An HTTP attempt has its own `attempt_id` and one-based `attempt_index` under the logical model call. One node can have one logical call with multiple SDK HTTP attempts. Failed requests without response headers carry a null completed duration and a client-observed upper bound. Early exit records the stage and agent whose output is reused.

| Metric | Unit and scope |
| --- | --- |
| `sample_wall_seconds` | Seconds from sample entry through graph execution and final scoring. Queue and export time are excluded. |
| `task_seconds`, `graph_seconds` | Seconds spent in the fixed or LAS graph. These cover the same interval in this project. Inline candidate grading supplies signals to the graph and is included here. |
| `scoring_seconds`, `final_base_evaluation_seconds` | Seconds in final scoring outside the graph and in the final candidate's base evaluation. The base evaluation can be reused from inside the graph. EvalPlus reference preparation is measured separately during dataset setup. |
| `process_cpu_seconds` | Parent-process CPU seconds over the sample interval. Concurrent samples can share this measurement. Child grading CPU is outside this scope. |
| `event_sink_seconds` | Seconds observed writing the sample's live events and finished spans, including waiting for the shared sink. |
| `inclusive_seconds`, `exclusive_seconds` | Function wall seconds, and the same interval minus the union of direct child intervals. Incomplete spans contribute only an observed interval. |
| `wall_sec`, `completion_tokens_per_call_second` | Logical client-call seconds and its provider-reported completion tokens divided by that duration. The synchronous clients leave TTFT unknown. The Ark adapter delegates output limits and temperature to the provider, so the call parameters for those values are null. |
| `known_*_tokens`, `known_api_fees` | Sums across calls with reported values. The matching `*_covered_calls` fields give the denominator. A null total means no measured values. Legacy stage and summary token columns may include estimates. |
| `http_attempts`, `http_covered_calls`, `http_retries` | Observed HTTP attempt count, calls with at least one observed attempt, and observed additional attempts beyond the first per call. |
| `accuracy`, `benchmark_scores` | Correct tasks divided by all committed tasks. EvalPlus base and combined plus pass rates use the same denominator. |
| `observed_samples_per_second` | Sample timing count divided by the sum of measured sample wall seconds. |
| Distribution columns | `count`, `sum`, `mean`, population `std`, `min`, `max`, `p50`, `p90`, `p95`, `p99`. Missing and nonfinite values are excluded. |

Per-model and per-role `tasks_correct_when_used` count correct final outcomes among unique tasks in which the model or role made a call. Node counts distinguish allocations and observed executions. Per-task trace timestamps and request timing come from the client. Server/GPU metrics and provider fees are left unavailable unless directly reported. Literal configured secrets are redacted across live and committed outputs. Prompts and responses are logged as experiment data.

## Workbook sheets

| Sheet | Row and columns |
| --- | --- |
| `tasks` | A committed task. Includes item and attempt IDs, query, raw response, reference and extracted code, entry point, extraction and execution status, correctness, base and additional test fields, route, sample timings, calls, and token coverage. |
| `agents` | A task's allocated debate worker or observed graph node. Includes model, final-node designation, observed executions, calls, failures, token/cost coverage, and call timing distribution. |
| `calls_N` | A logical model call. Includes request and response, linked node and span fields, backend, endpoint fingerprint, model IDs, finish reason, diagnostics, timings, `usage.*`, `provider_usage.*`, and `parameter.*`. |
| `events_N`, `rounds_N` | Events from committed attempts, and the debate rounds with their observed message links. |
| `spans_N`, `http_attempts_N` | Function and HTTP observations with join IDs, timings, status, and flattened `attributes.*`. |
| `model_metrics`, `role_metrics` | Per-model or agent-role aggregate counts, token and attempt coverage, and call timing. |
| `function_metrics`, `latency_metrics` | Per-function inclusive and exclusive duration distributions and status counts, and distributions by timing scope. |
| `run_summary` | Run status, experiment metadata, and summary metrics as field/value rows. |

Numbered sheets roll over before Excel's row limit. Formula-like prompt and response values are escaped, and long values are trimmed to Excel's cell limit. The JSON files retain the full redacted values.

The repository has evaluation workflows for MBPP, MBPP+, HumanEval, and HumanEval+. Training and benchmark-specific MasRouter analysis outputs apply to their own runners.
