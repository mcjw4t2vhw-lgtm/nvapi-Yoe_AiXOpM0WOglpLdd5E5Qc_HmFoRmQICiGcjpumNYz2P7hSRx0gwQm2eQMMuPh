"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from nvapi.errors import NvApiError
from nvapi.service import NvApi


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nvapi", description="Query GPU inventory")
    parser.add_argument(
        "--backend",
        choices=("auto", "mock", "nvidia-smi"),
        default="auto",
        help="Data source (default: auto)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List GPUs")
    get_cmd = sub.add_parser("get", help="Show one GPU")
    get_cmd.add_argument("index", type=int)
    sub.add_parser("summary", help="Show inventory summary")
    sub.add_parser("driver", help="Show driver info")
    return parser


def _make_api(backend: str) -> NvApi:
    if backend == "mock":
        return NvApi.mock()
    if backend == "nvidia-smi":
        return NvApi.nvidia_smi()
    return NvApi.auto()


def run(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        api = _make_api(args.backend)
        if args.command == "list":
            payload = [gpu.model_dump() for gpu in api.list_gpus()]
        elif args.command == "get":
            payload = api.get_gpu(args.index).model_dump()
        elif args.command == "summary":
            payload = api.summary().model_dump()
        else:
            payload = api.driver().model_dump()
    except NvApiError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2))
    return 0


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
