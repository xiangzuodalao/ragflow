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


SINGLE_CHUNK_CATEGORIES = [
    "fact",
    "procedure",
    "configuration",
    "troubleshooting",
    "comparison_or_boundary",
]
FULL_TROUBLESHOOTING_CATEGORY = "full_troubleshooting"
DEFAULT_CATEGORIES = [*SINGLE_CHUNK_CATEGORIES, FULL_TROUBLESHOOTING_CATEGORY]

STRUCTURED_FIELD_NAMES = [
    "车间",
    "产线",
    "设备编码",
    "设备名称",
    "故障类别",
    "故障代码",
    "故障描述",
    "步骤序号",
    "步骤内容",
    "维修方法",
    "判定标准",
    "标准时间",
]
STRUCTURED_FIELD_PATTERN = re.compile(
    r"(?:^|\s)-\s*(" + "|".join(re.escape(name) for name in STRUCTURED_FIELD_NAMES) + r")\s*[:：]\s*"
)


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


@dataclass
class ProcedureBundle:
    dataset_id: str
    document_ids: list[str]
    chunk_ids: list[str]
    document_names: list[str]
    chunks: list[CandidateChunk]
    fields: dict[str, str]
    steps: list[dict[str, str]]


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


def parse_structured_fields(text: str) -> dict[str, str]:
    matches = list(STRUCTURED_FIELD_PATTERN.finditer(text))
    fields: dict[str, str] = {}
    for idx, match in enumerate(matches):
        key = match.group(1)
        value_start = match.end()
        value_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        value = text[value_start:value_end].strip()
        fields[key] = value
    return fields


def parse_step_number(value: str) -> int | None:
    match = re.search(r"\d+", value or "")
    if not match:
        return None
    return int(match.group(0))


def strip_step_prefix(value: str) -> str:
    return re.sub(r"^\s*\d+\s*[.。)、:：-]?\s*", "", value or "").strip()


def normalize_group_value(value: str) -> str:
    return re.sub(r"\s+", "", value or "").lower()


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
    return candidates


def make_fault_group_key(fields: dict[str, str]) -> tuple[str, ...] | None:
    fault = fields.get("故障描述", "").strip()
    device_code = fields.get("设备编码", "").strip()
    step_no = parse_step_number(fields.get("步骤序号", ""))
    if not fault or not device_code or step_no is None:
        return None
    return (
        normalize_group_value(device_code),
        normalize_group_value(fault),
    )


def build_procedure_bundles(candidates: list[CandidateChunk], min_steps: int) -> list[ProcedureBundle]:
    grouped: dict[tuple[str, ...], list[tuple[CandidateChunk, dict[str, str], int]]] = {}
    for chunk in candidates:
        fields = parse_structured_fields(chunk.content)
        key = make_fault_group_key(fields)
        if key is None:
            continue
        step_no = parse_step_number(fields.get("步骤序号", ""))
        if step_no is None:
            continue
        grouped.setdefault(key, []).append((chunk, fields, step_no))

    bundles: list[ProcedureBundle] = []
    for group in grouped.values():
        ordered = sorted(group, key=lambda item: (item[2], item[0].document_name, item[0].chunk_id))
        if len({item[2] for item in ordered}) < min_steps:
            continue

        chunks = [item[0] for item in ordered]
        steps = [item[1] for item in ordered]
        first_fields = steps[0]
        bundles.append(
            ProcedureBundle(
                dataset_id=chunks[0].dataset_id,
                document_ids=list(dict.fromkeys(chunk.document_id for chunk in chunks)),
                chunk_ids=[chunk.chunk_id for chunk in chunks],
                document_names=list(dict.fromkeys(chunk.document_name for chunk in chunks)),
                chunks=chunks,
                fields=first_fields,
                steps=steps,
            )
        )
    bundles.sort(key=lambda bundle: (len(bundle.steps), len(" ".join(chunk.content for chunk in bundle.chunks))), reverse=True)
    return bundles


def build_full_troubleshooting_question(bundle: ProcedureBundle) -> str:
    fields = bundle.fields
    line = fields.get("产线", "").strip()
    device_name = fields.get("设备名称", "").strip()
    device_code = fields.get("设备编码", "").strip()
    fault_code = fields.get("故障代码", "").strip()
    fault = fields.get("故障描述", "").strip()
    device = device_name or device_code or "设备"
    location = f"{line}产线的" if line else ""
    code_part = f"（设备编码{device_code}）" if device_code and device_code != device else ""
    fault_part = f"{fault_code}，{fault}" if fault_code else fault
    return f"{location}{device}{code_part}出现“{fault_part}”时怎么处理？请按完整步骤说明。"


def build_full_troubleshooting_ground_truth(bundle: ProcedureBundle) -> str:
    fields = bundle.fields
    device_name = fields.get("设备名称", "").strip()
    device_code = fields.get("设备编码", "").strip()
    fault_code = fields.get("故障代码", "").strip()
    fault = fields.get("故障描述", "").strip()
    device = device_name or device_code or "该设备"
    code_part = f"（设备编码{device_code}）" if device_code and device_code != device else ""
    fault_part = f"{fault_code}，{fault}" if fault_code else fault
    lines = [f"{device}{code_part}出现“{fault_part}”时，按以下步骤处理："]
    for idx, step in enumerate(bundle.steps, start=1):
        step_no = parse_step_number(step.get("步骤序号", "")) or idx
        content = strip_step_prefix(step.get("步骤内容", ""))
        method = strip_step_prefix(step.get("维修方法", ""))
        criterion = strip_step_prefix(step.get("判定标准", ""))
        standard_time = step.get("标准时间", "").strip()

        parts = []
        if content:
            parts.append(content)
        if method and method != content:
            parts.append(f"维修方法：{method}")
        if criterion and criterion not in {content, method, "其它", "NA", "N/A"}:
            parts.append(f"判定标准：{criterion}")
        if standard_time:
            parts.append(f"标准时间：{standard_time}")
        if not parts:
            parts.append(bundle.chunks[idx - 1].content)
        lines.append(f"{step_no}. " + "；".join(parts))
    return "\n".join(lines)


def build_full_troubleshooting_sample(bundle: ProcedureBundle, sample_id: str, include_ground_truth: bool) -> dict[str, Any]:
    evidence = "\n".join(chunk.content for chunk in bundle.chunks)
    return {
        "id": sample_id,
        "question": build_full_troubleshooting_question(bundle),
        "ground_truth": build_full_troubleshooting_ground_truth(bundle) if include_ground_truth else "",
        "expected_evidence": evidence,
        "source_chunk_id": bundle.chunk_ids[0],
        "source_chunk_ids": bundle.chunk_ids,
        "source_document_id": bundle.document_ids[0],
        "source_document_ids": bundle.document_ids,
        "document_name": bundle.document_names[0],
        "document_names": bundle.document_names,
        "dataset_id": bundle.dataset_id,
        "category": FULL_TROUBLESHOOTING_CATEGORY,
        "difficulty": "hard" if len(bundle.steps) >= 4 else "medium",
        "evidence_type": "procedure_bundle",
        "procedure_step_count": len(bundle.steps),
    }


def determine_full_troubleshooting_count(sample_size: int, explicit_count: int, ratio: float) -> int:
    if explicit_count >= 0:
        return max(0, min(explicit_count, sample_size))
    if sample_size <= 0:
        return 0
    return max(1, min(sample_size, round(sample_size * ratio))) if ratio > 0 else 0


def interleave_samples(singles: list[CandidateChunk], bundles: list[ProcedureBundle]) -> list[CandidateChunk | ProcedureBundle]:
    items: list[CandidateChunk | ProcedureBundle] = []
    single_idx = 0
    bundle_idx = 0
    while single_idx < len(singles) or bundle_idx < len(bundles):
        if bundle_idx < len(bundles) and (len(items) % 3 == 0 or single_idx >= len(singles)):
            items.append(bundles[bundle_idx])
            bundle_idx += 1
        elif single_idx < len(singles):
            items.append(singles[single_idx])
            single_idx += 1
        else:
            items.append(bundles[bundle_idx])
            bundle_idx += 1
    return items


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


def generate_ground_truth(client: Any, model: str, question: str, evidence: str, max_source_chars: int) -> str:
    source = evidence[:max_source_chars]
    prompt = (
        "Write a concise ground-truth answer for the question using only the source evidence. "
        "Use the same language as the question. "
        "Do not mention that the answer comes from source evidence. "
        "If the evidence contains structured fields, preserve the exact field values needed to answer. "
        "Return strict JSON with key: ground_truth.\n\n"
        f"Question:\n{question}\n\n"
        f"Source evidence:\n{source}"
    )
    messages = [
        {"role": "system", "content": "You create grounded reference answers for RAG evaluation."},
        {"role": "user", "content": prompt},
    ]
    kwargs = {"model": model, "messages": messages, "temperature": 0}
    try:
        completion = client.chat.completions.create(response_format={"type": "json_object"}, **kwargs)
        obj = extract_json_object(completion.choices[0].message.content or "")
        ground_truth = str(obj.get("ground_truth", "")).strip()
    except Exception:
        completion = client.chat.completions.create(**kwargs)
        text = completion.choices[0].message.content or ""
        try:
            obj = extract_json_object(text)
            ground_truth = str(obj.get("ground_truth", "")).strip()
        except Exception:
            ground_truth = text.strip()
    if not ground_truth:
        raise ValueError("Ground-truth generator returned an empty answer.")
    return ground_truth


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
    parser.add_argument("--skip-ground-truth", action="store_true")
    parser.add_argument(
        "--full-troubleshooting-count",
        type=int,
        default=-1,
        help="Number of full troubleshooting samples. -1 derives the count from --full-troubleshooting-ratio.",
    )
    parser.add_argument("--full-troubleshooting-ratio", type=float, default=0.35)
    parser.add_argument("--full-troubleshooting-min-steps", type=int, default=2)
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
        min_chars=args.min_chars,
        max_documents=args.max_documents,
        seed=args.seed,
    )

    rng = random.Random(args.seed)
    full_count = determine_full_troubleshooting_count(
        args.sample_size,
        args.full_troubleshooting_count,
        args.full_troubleshooting_ratio,
    )
    bundles = build_procedure_bundles(candidates, args.full_troubleshooting_min_steps)
    rng.shuffle(bundles)
    selected_bundles = bundles[:full_count]
    used_chunk_ids = {chunk_id for bundle in selected_bundles for chunk_id in bundle.chunk_ids}
    single_count = args.sample_size - len(selected_bundles)
    single_pool = [chunk for chunk in candidates if chunk.chunk_id not in used_chunk_ids]
    rng.shuffle(single_pool)
    selected_singles = single_pool[:single_count]
    items = interleave_samples(selected_singles, selected_bundles)
    if len(items) < args.sample_size:
        print(
            f"Warning: requested {args.sample_size} samples but only {len(items)} usable samples were available.",
            file=sys.stderr,
        )
    if full_count and len(selected_bundles) < full_count:
        print(
            f"Warning: requested {full_count} full troubleshooting samples but only found {len(selected_bundles)} bundles.",
            file=sys.stderr,
        )

    judge = get_openai_client()
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(items, start=1):
        sample_id = f"{args.run_id}-{idx:04d}"
        if isinstance(item, ProcedureBundle):
            row = build_full_troubleshooting_sample(item, sample_id, include_ground_truth=not args.skip_ground_truth)
            rows.append(row)
            print(
                f"[{idx}/{len(items)}] {row['category']} ({row['procedure_step_count']} steps): {row['question']}"
            )
            continue

        chunk = item
        category = SINGLE_CHUNK_CATEGORIES[(idx - 1) % len(SINGLE_CHUNK_CATEGORIES)]
        generated = generate_question(judge, args.model, chunk, category, args.max_source_chars)
        ground_truth = ""
        if not args.skip_ground_truth:
            ground_truth = generate_ground_truth(judge, args.model, generated["question"], chunk.content, args.max_source_chars)
        rows.append(
            {
                "id": sample_id,
                "question": generated["question"],
                "ground_truth": ground_truth,
                "expected_evidence": chunk.content,
                "source_chunk_id": chunk.chunk_id,
                "source_document_id": chunk.document_id,
                "document_name": chunk.document_name,
                "dataset_id": chunk.dataset_id,
                "category": generated["category"],
                "difficulty": generated["difficulty"],
                "evidence_type": "single_chunk",
            }
        )
        print(f"[{idx}/{len(items)}] {generated['category']}: {generated['question']}")

    write_jsonl(output, rows)
    print(f"Wrote {len(rows)} eval samples to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
