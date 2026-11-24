from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import yaml

from las_mbpp.config import PROJECT_ROOT
from las_mbpp.keys import load_env_file


def _load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


def _workflow_main(name: str) -> Callable[[], None]:
    normalized = name.strip().lower().replace("-", "_")
    if normalized in {"las", "las_distilbert", "mbpp_las", "mbpp_las_distilbert"}:
        from las_mbpp.workflows.mbpp_las_distilbert import main

        return main
    if normalized in {"baseline", "fixed", "mbpp_baseline"}:
        from las_mbpp.workflows.mbpp_baseline import main

        return main
    raise ValueError(f"Unknown workflow: {name}")


def _value_to_cli(key: str, value: Any) -> list[str]:
    flag = f"--{key}"
    if value is None or value is False:
        return []
    if value is True:
        return [flag]
    if isinstance(value, (list, tuple, set)):
        return [flag, ",".join(str(item) for item in value)]
    return [flag, str(value)]


def _args_to_cli(raw_args: dict[str, Any]) -> list[str]:
    cli_args: list[str] = []
    for key, value in raw_args.items():
        cli_args.extend(_value_to_cli(str(key), value))
    return cli_args


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run an MBPP workflow from a YAML or JSON config.")
    parser.add_argument("config", type=Path, help="Path to a config file under configs/ or an absolute path.")
    parser.add_argument("--env-file", type=Path, default=None, help="Optional dotenv file. Defaults to config env_file or .env.")
    return parser.parse_known_args()


def main() -> None:
    args, extra_args = parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = _load_config(config_path)
    env_file = args.env_file or config.get("env_file")
    load_env_file(env_file)

    workflow = str(config.get("workflow", "las_distilbert"))
    workflow_args = config.get("args", {})
    if not isinstance(workflow_args, dict):
        raise ValueError("Config field 'args' must be a mapping.")

    cli_args = [*_args_to_cli(workflow_args), *extra_args]
    sys.argv = [f"las-mbpp:{workflow}", *cli_args]
    _workflow_main(workflow)()


if __name__ == "__main__":
    main()
