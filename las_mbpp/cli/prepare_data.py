"""Populate the official cache used directly by the experiment runners."""

import argparse
from pathlib import Path
from urllib.request import urlopen

from datasets import load_dataset

from las_mbpp.benchmarks import (
    DATASETS, MBPP_RELEASE_URL, load_official_problems, read_mbpp_release,
)
from las_mbpp.run_artifacts import atomic_write


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--mbpp-json", type=Path, help="Download a pinned upstream sanitized JSON file for offline use.")
    args = parser.parse_args()
    if args.mbpp_json and args.dataset != "mbpp":
        parser.error("--mbpp-json applies to mbpp only.")
    if args.dataset == "mbpp":
        try:
            if args.mbpp_json:
                if not args.mbpp_json.exists():
                    def download(path):
                        with urlopen(MBPP_RELEASE_URL, timeout=30) as response:
                            path.write_bytes(response.read())
                        read_mbpp_release(path)

                    atomic_write(args.mbpp_json, download)
                dataset = read_mbpp_release(args.mbpp_json)
                print(f"Set MBPP_SANITIZED_JSON={args.mbpp_json.resolve()} for the runners.")
            else:
                dataset = load_dataset("google-research-datasets/mbpp", "sanitized")
        except (OSError, ValueError, ConnectionError) as exc:
            parser.error(f"Cannot prepare sanitized MBPP ({type(exc).__name__}). Check Hub access or use --mbpp-json PATH for the official Google Research copy.")
        print("MBPP splits:", ", ".join(f"{split} ({len(rows)})" for split, rows in dataset.items()))
        return
    try:
        problems, fingerprint = load_official_problems(args.dataset)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(f"{exc} Check network access or seed the official EvalPlus cache.")
    print(f"{args.dataset}: {len(problems)} official test problems cached.")
    if fingerprint:
        print(f"Official release fingerprint (MD5): {fingerprint}")


if __name__ == "__main__":
    main()
