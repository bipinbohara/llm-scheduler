#!/usr/bin/env bash
set -euo pipefail

python -m las_mbpp.cli.run_from_config configs/mbpp_baseline.yaml "$@"
