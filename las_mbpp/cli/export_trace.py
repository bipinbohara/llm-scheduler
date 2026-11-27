"""Build a Perfetto timeline from a live event or span journal."""

from __future__ import annotations

import argparse
from pathlib import Path

from las_mbpp.experiment_report import chrome_trace
from las_mbpp.journals import recover
from las_mbpp.run_artifacts import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journal", type=Path, help="Run events.jsonl or spans.jsonl")
    parser.add_argument("output", type=Path, help="Destination Chrome Trace Event JSON")
    args = parser.parse_args()
    if not args.journal.is_file() or args.journal.resolve() == args.output.resolve():
        parser.error("Supply an existing journal and a separate output file.")
    rows, tail_bytes = recover(args.journal)
    spans = {}
    for row in rows:
        if row.get("event") not in {"span_started", "span_finished", None}:
            continue
        if row.get("span_id") and row.get("started_at"):
            spans[row["span_id"]] = row
    write_json(args.output, chrome_trace(
        spans.values(), journal=str(args.journal), ignored_partial_tail_bytes=tail_bytes,
        unfinished_spans=sum(span.get("inclusive_seconds") is None for span in spans.values()),
    ))


if __name__ == "__main__":
    main()
