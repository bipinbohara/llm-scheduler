from __future__ import annotations

from las_mbpp.keys import load_env_file


def main() -> None:
    load_env_file()
    from las_mbpp.workflows.mbpp_las_distilbert import main as workflow_main

    workflow_main()


if __name__ == "__main__":
    main()
