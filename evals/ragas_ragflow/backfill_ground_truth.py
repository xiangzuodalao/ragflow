#!/usr/bin/env python3
"""Backfill generated ground-truth answers into an existing evalset JSONL."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from generate_evalset import generate_ground_truth, get_openai_client, load_env_file


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evalset")
    parser.add_argument("--env-file", default=str(Path(__file__).with_name(".env")))
    parser.add_argument("--output", help="Defaults to overwriting the input evalset.")
    parser.add_argument("--model", default=os.environ.get("RAGAS_JUDGE_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--max-source-chars", type=int, default=3000)
    parser.add_argument("--force", action="store_true", help="Regenerate existing ground_truth values.")
    return parser


def main() -> int:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--env-file", default=str(Path(__file__).with_name(".env")))
    pre_args, _ = pre_parser.parse_known_args()
    load_env_file(pre_args.env_file)

    parser = build_arg_parser()
    args = parser.parse_args()
    load_env_file(args.env_file)

    input_path = Path(args.evalset)
    output_path = Path(args.output) if args.output else input_path
    rows = read_jsonl(input_path)
    client = get_openai_client()

    changed = 0
    for idx, row in enumerate(rows, start=1):
        if row.get("ground_truth") and not args.force:
            continue
        ground_truth = generate_ground_truth(
            client,
            args.model,
            row["question"],
            row.get("expected_evidence", ""),
            args.max_source_chars,
        )
        row["ground_truth"] = ground_truth
        changed += 1
        print(f"[{idx}/{len(rows)}] ground_truth_len={len(ground_truth)}", flush=True)

    write_jsonl(output_path, rows)
    print(f"Wrote {len(rows)} rows to {output_path}; changed={changed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
