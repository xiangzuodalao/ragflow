#!/usr/bin/env python3
"""Generate a Ragas evaluation set from real RAGFlow chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CATEGORIES = [
    "fact",
    "procedure",
    "configuration",
    "troubleshooting",
    "comparison_or_boundary",
]


def load_env_file(path: str | None) -> None:
    if not path:
        return
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = os.path.expandvars(value.strip().strip('"').strip("'"))
        os.environ.setdefault(key.strip(), value)


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def require_env(name: str) -> str:
    value = env(name)
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def normalize_base_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/api/v1"):
        return base_url[:-7]
    return base_url


class RAGFlowClient:
    def __init__(self, base_url: str, api_key: str) -> None:
        try:
            import requests
        except ImportError as exc:
            raise SystemExit(
                "Missing dependency 'requests'. Install eval deps with: "
                "uv pip install requests openai ragas datasets langchain-openai pandas"
            ) from exc
        self.base_url = normalize_base_url(base_url)
        self.api_url = f"{self.base_url}/api/v1"
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def request(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self.session.request(method, f"{self.api_url}{path}", timeout=120, **kwargs)
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != 0:
            raise RuntimeError(f"RAGFlow API error on {method} {path}: {payload.get('message')}")
        return payload.get("data")

    def list_datasets(self) -> list[dict[str, Any]]:
        datasets: list[dict[str, Any]] = []
        page = 1
        while True:
            data = self.request("GET", "/datasets", params={"page": page, "page_size": 100})
            batch = data if isinstance(data, list) else data.get("datasets", [])
            if not batch:
                break
            datasets.extend(batch)
            if len(batch) < 100:
                break
            page += 1
        return datasets

    def list_documents(self, dataset_id: str) -> list[dict[str, Any]]:
        docs: list[dict[str, Any]] = []
        page = 1
        while True:
            data = self.request(
                "GET",
                f"/datasets/{dataset_id}/documents",
                params={"page": page, "page_size": 100, "orderby": "create_time", "desc": True},
            )
            batch = data.get("docs", []) if isinstance(data, dict) else []
            if not batch:
                break
            docs.extend(batch)
            total = int(data.get("total", 0) or 0)
            if len(docs) >= total or len(batch) < 100:
                break
            page += 1
        return docs

    def list_chunks(self, dataset_id: str, document_id: str, page_size: int = 100) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        page = 1
        while True:
            data = self.request(
                "GET",
                f"/datasets/{dataset_id}/documents/{document_id}/chunks",
                params={"page": page, "page_size": page_size},
            )
            batch = data.get("chunks", []) if isinstance(data, dict) else []
            if not batch:
                break
            chunks.extend(batch)
            total = int(data.get("total", 0) or 0)
            if len(chunks) >= total or len(batch) < page_size:
                break
            page += 1
        return chunks


@dataclass
class CandidateChunk:
    dataset_id: str
    document_id: str
    chunk_id: str
    document_name: str
    content: str


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_noisy_chunk(text: str, min_chars: int) -> bool:
    if len(text) < min_chars:
        return True
    if len(set(text)) < 16:
        return True
    pipe_ratio = text.count("|") / max(len(text), 1)
    tab_ratio = text.count("\t") / max(len(text), 1)
    digit_ratio = sum(ch.isdigit() for ch in text) / max(len(text), 1)
    return pipe_ratio > 0.08 or tab_ratio > 0.05 or digit_ratio > 0.55


def stable_hash(text: str) -> str:
    normalized = re.sub(r"\W+", "", text.lower())[:600]
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def choose_dataset(client: RAGFlowClient, explicit_dataset_id: str | None) -> dict[str, Any]:
    datasets = client.list_datasets()
    if explicit_dataset_id:
        for dataset in datasets:
            if dataset.get("id") == explicit_dataset_id:
                return dataset
        raise SystemExit(f"Dataset not found or inaccessible: {explicit_dataset_id}")
    valid = [
        dataset
        for dataset in datasets
        if int(dataset.get("chunk_count") or dataset.get("chunk_num") or 0) > 0
    ]
    if not valid:
        raise SystemExit("No parsed dataset with chunks was found for this API key.")
    valid.sort(
        key=lambda item: (
            int(item.get("chunk_count") or item.get("chunk_num") or 0),
            int(item.get("document_count") or item.get("document_num") or 0),
        ),
        reverse=True,
    )
    return valid[0]


def collect_candidates(
    client: RAGFlowClient,
    dataset_id: str,
    sample_size: int,
    min_chars: int,
    max_documents: int,
    seed: int,
) -> list[CandidateChunk]:
    docs = client.list_documents(dataset_id)
    parsed_docs = [
        doc for doc in docs
        if int(doc.get("chunk_count") or doc.get("chunk_num") or 0) > 0
    ]
    if max_documents > 0:
        parsed_docs = parsed_docs[:max_documents]
    if not parsed_docs:
        raise SystemExit(f"No parsed documents with chunks found in dataset {dataset_id}.")

    candidates: list[CandidateChunk] = []
    seen: set[str] = set()
    for doc in parsed_docs:
        doc_id = doc.get("id")
        if not doc_id:
            continue
        doc_name = doc.get("name") or doc.get("document_name") or doc_id
        for chunk in client.list_chunks(dataset_id, doc_id):
            content = clean_text(chunk.get("content", ""))
            if is_noisy_chunk(content, min_chars):
                continue
            fingerprint = stable_hash(content)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            candidates.append(
                CandidateChunk(
                    dataset_id=dataset_id,
                    document_id=doc_id,
                    chunk_id=chunk.get("id", ""),
                    document_name=doc_name,
                    content=content,
                )
            )

    if not candidates:
        raise SystemExit("No usable chunks remained after filtering.")
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[: min(sample_size, len(candidates))]


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def get_openai_client() -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'openai'. Install eval deps with: "
            "uv pip install requests openai ragas datasets langchain-openai pandas"
        ) from exc
    base_url = env("DASHSCOPE_BASE_URL")
    if not base_url:
        workspace_id = require_env("DASHSCOPE_WORKSPACE_ID")
        base_url = f"https://{workspace_id}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    return OpenAI(api_key=require_env("DASHSCOPE_API_KEY"), base_url=base_url)


def generate_question(client: Any, model: str, chunk: CandidateChunk, category: str, max_source_chars: int) -> dict[str, str]:
    source = chunk.content[:max_source_chars]
    prompt = (
        "Rewrite the source evidence into one realistic end-user question for RAG evaluation. "
        "The question must be answerable from the source evidence alone. "
        "Use the same language as the source evidence. "
        f"Target category: {category}. "
        "Return strict JSON with keys: question, category, difficulty. "
        "difficulty must be one of easy, medium, hard.\n\n"
        f"Source evidence:\n{source}"
    )
    messages = [
        {"role": "system", "content": "You create concise, grounded RAG evaluation questions."},
        {"role": "user", "content": prompt},
    ]
    kwargs = {"model": model, "messages": messages, "temperature": 0.2}
    try:
        completion = client.chat.completions.create(response_format={"type": "json_object"}, **kwargs)
    except Exception:
        completion = client.chat.completions.create(**kwargs)
    obj = extract_json_object(completion.choices[0].message.content or "")
    question = str(obj.get("question", "")).strip()
    if not question:
        raise ValueError("Question generator returned an empty question.")
    generated_category = str(obj.get("category") or category).strip() or category
    if generated_category not in DEFAULT_CATEGORIES:
        generated_category = category
    return {
        "question": question,
        "category": generated_category,
        "difficulty": str(obj.get("difficulty") or "medium").strip() or "medium",
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=str(Path(__file__).with_name(".env")))
    parser.add_argument("--dataset-id")
    parser.add_argument("--sample-size", type=int, default=int(env("RAGAS_EVAL_SAMPLE_SIZE", "50") or "50"))
    parser.add_argument("--min-chars", type=int, default=120)
    parser.add_argument("--max-source-chars", type=int, default=3000)
    parser.add_argument("--max-documents", type=int, default=0, help="0 means no document limit.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output")
    parser.add_argument("--run-id", default=time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--model", default=env("RAGAS_JUDGE_MODEL", "deepseek-v4-flash"))
    return parser


def main() -> int:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--env-file", default=str(Path(__file__).with_name(".env")))
    pre_args, _ = pre_parser.parse_known_args()
    load_env_file(pre_args.env_file)

    parser = build_arg_parser()
    args = parser.parse_args()
    load_env_file(args.env_file)

    output = Path(args.output) if args.output else Path("evals/ragas_ragflow/outputs") / args.run_id / "evalset.jsonl"
    ragflow = RAGFlowClient(require_env("RAGFLOW_BASE_URL"), require_env("RAGFLOW_API_KEY"))
    dataset = choose_dataset(ragflow, args.dataset_id)
    candidates = collect_candidates(
        ragflow,
        dataset_id=dataset["id"],
        sample_size=args.sample_size,
        min_chars=args.min_chars,
        max_documents=args.max_documents,
        seed=args.seed,
    )

    judge = get_openai_client()
    rows: list[dict[str, Any]] = []
    for idx, chunk in enumerate(candidates, start=1):
        category = DEFAULT_CATEGORIES[(idx - 1) % len(DEFAULT_CATEGORIES)]
        generated = generate_question(judge, args.model, chunk, category, args.max_source_chars)
        rows.append(
            {
                "id": f"{args.run_id}-{idx:04d}",
                "question": generated["question"],
                "expected_evidence": chunk.content,
                "source_chunk_id": chunk.chunk_id,
                "source_document_id": chunk.document_id,
                "document_name": chunk.document_name,
                "dataset_id": chunk.dataset_id,
                "category": generated["category"],
                "difficulty": generated["difficulty"],
            }
        )
        print(f"[{idx}/{len(candidates)}] {generated['category']}: {generated['question']}")

    write_jsonl(output, rows)
    print(f"Wrote {len(rows)} eval samples to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
