# Run artifacts and workbook fields

Each run has one configuration and one line-oriented `run.log` for start, resume, task outcome, checkpoint load, and finish events. The console prints the same events. Per-task JSON saves the description, reference answer, final evaluator output, stage rows, and structured call records. Stage CSVs expose the existing prompt, response, generation token, model, route, correctness, and timing fields. JSON and CSV are the full-text exports.

Call IDs are local to the task and include a component prefix. Each call records its request, text response, UTC start, wall time, outcome, provider completion status if exposed, and token usage. Matched calls carry the stage, agent, and attribution. Debate revisions are collaboration, consensus is aggregation, the LAS call is routing, and reflection, verification, and repair are extra processing. The same debate worker runs the initial and revision rounds. A local provider call records one HTTP attempt and zero retries. Cloud SDKs manage internal retries, so attempt and retry counts are unknown for cloud calls. Provider cost data is unavailable. No cost estimate is reported.

The fixed workflow allocates A, B, and C and runs two rounds. For LAS, stage rows give the participating workers and completed rounds before routing. The terminal stage identifies the final output node. A normal output comes from the correction agent and its model. For early exit, the output node's JSON identifies the source agent and stage, which can be joined to the producing stage's model. The JSON execution object records the allocated and participating workers, observed rounds, distinct nodes with any assigned models, stage invocation count, and final source. The LAS router counts as an observed node if called. Gate scoring is a separate timing component. A stage row is a stage invocation rather than a distinct worker. The local tester and early-exit forwarding node have no model call.

## `summary.xlsx`

Enable with `--excel`. One worksheet has one row per completed task, in saved item order. The schema also describes `summary.csv`. The workbook sanitizes text for Excel and truncates cells to Excel's length limit. Use JSON for full source and trace content.

| Field | Unit | Meaning and scope |
| --- | --- | --- |
| `run_id` | ID | Existing run identifier, repeated per task. |
| `dataset` | name | Benchmark adapter for the run. New. |
| `split` | name | Existing official split name. |
| `task_id` | ID | Existing numeric task identifier within the split. |
| `benchmark_id` | ID | New original namespaced ID when supplied by the adapter. |
| `function_name_from_tests` | name | Existing requested entry point of the task. |
| `final_is_correct` | boolean | Existing final correctness. EvalPlus requires both suites to pass. |
| `outcome` | category | New task outcome: correct, incorrect, or invalid (for example, empty output or a pipeline error). |
| `tests_passed` | count | Existing passed native assertions or EvalPlus base inputs on the final candidate. Original HumanEval counts its check function as one suite. |
| `tests_total` | count | Existing total assertions or base inputs in the final evaluation. |
| `final_exec_status` | text | Existing status of the native or base final evaluation. |
| `final_code` | Python text | Existing final candidate submitted to scoring. |
| `final_route_action` | category | Existing LAS route applied to the final output. Blank when no LAS shortcut was applied. |
| `task_wall_sec` | seconds | New elapsed worker time from pipeline entry through final scoring, including overlap within the task. Excludes queue time and exports. |
| `logical_calls` | count | New count of model client invocations for the task, including LAS calls and failures. |
| `known_attempts` | count | New sum of observed HTTP attempts across calls with available attempt counts. |
| `calls_with_unknown_attempts` | count | New count of logical calls whose transport attempts were unobservable. |
| `input_tokens` | tokens | New task sum of existing call input token counts. Cloud clients may estimate missing usage. Excludes gate tokenizer counts. |
| `output_tokens` | tokens | New task sum of existing call output token counts, with the same usage scope. |
| `total_tokens` | tokens | New task sum of existing call total token counts, with the same usage scope. |
| `task_csv` | path | Existing stage CSV path. |
| `evalplus_base_correct` | boolean | New EvalPlus-only final base-suite pass. |
| `evalplus_plus_correct` | boolean | New EvalPlus-only final combined base and additional-suite pass. |
| `plus_tests_passed` | count | New EvalPlus-only passed additional inputs. |
| `plus_tests_total` | count | New EvalPlus-only official additional inputs. |
| `plus_exec_status` | text | New EvalPlus-only status of the additional suite. |
| `plus_latency_sec` | seconds | New wall time to evaluate the final candidate on the additional suite. Included in task wall time. |

In stage CSVs, `latency_sec` is the stage's model latency, except for the testing row where it is evaluator time. `eval_latency_sec` is time in the candidate evaluator and may be copied forward when reusing a score. `gate_latency_sec` includes local gate scoring and decision features. `las_latency_sec` is the separate router model call. The JSON call's `wall_sec` times the client invocation. Summing call, gate, or stage times does not yield task wall time when work overlaps or stage results are reused.
