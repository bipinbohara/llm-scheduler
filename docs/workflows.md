# Workflows

## LAS DistilBERT Workflow

`las_mbpp.workflows.mbpp_las_distilbert` implements the MBPP LAS path from the paper:

1. Debate agents generate and revise candidate code.
2. Consensus selects a candidate.
3. Reflection and verification refine the selected code.
4. The local tester runs official MBPP tests.
5. Error correction repairs code from test feedback.
6. After each intermediate stage, the gate scores the artifact.
7. When the gate passes its threshold, the LAS scheduler chooses `early_exit`, `test`, or `refinement`.

## Fixed Baseline Workflow

`las_mbpp.workflows.mbpp_baseline` runs the same multi-stage pipeline without dynamic LAS routing. It is useful for smoke tests and for comparing route savings against the fixed full workflow.
