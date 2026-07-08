#!/usr/bin/env python3
"""Run a RAGFlow chat evaluation with Ragas."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any


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
        response = self.session.request(method, f"{self.api_url}{path}", timeout=180, **kwargs)
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != 0:
            raise RuntimeError(f"RAGFlow API error on {method} {path}: {payload.get('message')}")
        return payload.get("data")

    def create_chat(self, name: str, dataset_ids: list[str], args: argparse.Namespace) -> dict[str, Any]:
        prompt_config: dict[str, Any] = {"enable_table_entity_filter": not args.disable_table_entity_filter}
        if args.disable_sql_retrieval:
            prompt_config["disable_sql_retrieval"] = True
        payload = {
            "name": name,
            "dataset_ids": dataset_ids,
            "similarity_threshold": args.similarity_threshold,
            "vector_similarity_weight": args.vector_similarity_weight,
            "top_k": args.top_k,
            "top_n": args.top_n,
            "rerank_id": args.rerank_id,
            "prompt_config": prompt_config,
        }
        return self.request("POST", "/chats", json=payload)

    def create_session(self, chat_id: str, name: str) -> str:
        data = self.request("POST", f"/chats/{chat_id}/sessions", json={"name": name})
        return data["id"]

    def delete_sessions(self, chat_id: str, session_ids: list[str]) -> None:
        if session_ids:
            self.request("DELETE", f"/chats/{chat_id}/sessions", json={"ids": session_ids})

    def list_sessions(self, chat_id: str, page: int = 1, page_size: int = 100) -> list[dict[str, Any]]:
        data = self.request(
            "GET",
            f"/chats/{chat_id}/sessions",
            params={"page": page, "page_size": page_size, "orderby": "create_time", "desc": True},
        )
        return data if isinstance(data, list) else []

    def list_eval_sessions(self, chat_id: str, run_id: str) -> list[dict[str, Any]]:
        prefix = f"eval-{run_id}-"
        matches: list[dict[str, Any]] = []
        page = 1
        while True:
            batch = self.list_sessions(chat_id, page=page, page_size=100)
            if not batch:
                break
            matches.extend([session for session in batch if str(session.get("name", "")).startswith(prefix)])
            if len(batch) < 100:
                break
            page += 1
        return matches

    def ask(self, chat_id: str, session_id: str, question: str, args: argparse.Namespace) -> dict[str, Any]:
        # Pass switches per-request too, so retrieval routing is correct even if the
        # chat-stored prompt_config was stripped of non-standard keys.
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "session_id": session_id,
            "question": question,
            "stream": False,
            "enable_table_entity_filter": not args.disable_table_entity_filter,
        }
        if args.disable_sql_retrieval:
            payload["disable_sql_retrieval"] = True
        return self.request("POST", "/chat/completions", json=payload)

    def retrieve(
        self,
        question: str,
        dataset_ids: list[str],
        args: argparse.Namespace,
        reference_chunk_count: int = 0,
    ) -> dict[str, Any]:
        page_size = args.fallback_retrieval_page_size
        if page_size <= 0:
            page_size = reference_chunk_count or args.top_n
        if reference_chunk_count > 0:
            page_size = min(page_size, reference_chunk_count)
        payload = {
            "question": question,
            "dataset_ids": dataset_ids,
            "page": 1,
            "page_size": max(page_size, 1),
            "similarity_threshold": args.similarity_threshold,
            "vector_similarity_weight": args.vector_similarity_weight,
            "top_k": args.top_k,
            "rerank_id": args.rerank_id,
            "highlight": False,
            "use_kg": False,
            "cross_languages": [],
            "enable_table_entity_filter": not args.disable_table_entity_filter,
        }
        return self.request("POST", "/retrieval", json=payload)


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


CHUNK_TEXT_FIELDS = (
    "content",
    "content_with_weight",
    "content_with_weight_ltks",
    "text",
    "highlight",
)


def normalize_context_text(value: Any) -> str:
    if isinstance(value, str):
        text = value
    elif isinstance(value, list):
        text = "\n".join(str(item) for item in value if item is not None)
    else:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_context_text(chunk: dict[str, Any]) -> tuple[str, str]:
    for field in CHUNK_TEXT_FIELDS:
        text = normalize_context_text(chunk.get(field))
        if text:
            return text, field
    return "", ""


def contexts_from_reference(reference: Any) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    stats: dict[str, Any] = {
        "reference_chunk_count": 0,
        "extracted_context_count": 0,
        "null_content_chunk_count": 0,
        "empty_text_chunk_count": 0,
        "field_counts": {},
        "extraction_source": "reference",
    }
    if not isinstance(reference, dict):
        stats["extraction_source"] = "none"
        return [], [], stats
    contexts: list[str] = []
    raw_chunks: list[dict[str, Any]] = []
    for chunk in reference.get("chunks") or []:
        if not isinstance(chunk, dict):
            continue
        raw_chunks.append(chunk)
        if chunk.get("content") is None:
            stats["null_content_chunk_count"] += 1
        text, field = chunk_context_text(chunk)
        if text:
            contexts.append(text)
            field_counts = stats["field_counts"]
            field_counts[field] = field_counts.get(field, 0) + 1
        else:
            stats["empty_text_chunk_count"] += 1
    stats["reference_chunk_count"] = len(raw_chunks)
    stats["extracted_context_count"] = len(contexts)
    if not contexts and raw_chunks:
        stats["extraction_source"] = "empty_reference_chunks"
    return contexts, raw_chunks, stats


def contexts_from_prompt(prompt: Any) -> list[str]:
    if not isinstance(prompt, str) or "Content:" not in prompt:
        return []
    contexts: list[str] = []
    seen: set[str] = set()
    blocks = re.split(r"\n\s*------\s*\n", prompt)
    for block in blocks:
        if "Content:" not in block:
            continue
        _, content = block.split("Content:", 1)
        content = re.split(
            r"\n\s*(?:------|The above is the knowledge base\.|### Query:|## Time elapsed:)",
            content,
            maxsplit=1,
        )[0]
        text = normalize_context_text(content)
        if not text or text in seen:
            continue
        seen.add(text)
        contexts.append(text)
    return contexts


def contexts_from_response(data: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    contexts, raw_chunks, stats = contexts_from_reference(data.get("reference"))
    if contexts:
        return contexts, raw_chunks, stats

    prompt_contexts = contexts_from_prompt(data.get("prompt"))
    if prompt_contexts:
        stats = dict(stats)
        stats["extraction_source"] = "prompt_fallback"
        stats["prompt_context_count"] = len(prompt_contexts)
        stats["extracted_context_count"] = len(prompt_contexts)
        return prompt_contexts, raw_chunks, stats

    stats["prompt_context_count"] = 0
    return contexts, raw_chunks, stats


def fallback_dataset_ids(sample: dict[str, Any], raw_chunks: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    sample_dataset_id = sample.get("dataset_id")
    if isinstance(sample_dataset_id, str) and sample_dataset_id.strip():
        ids.append(sample_dataset_id.strip())
    for chunk in raw_chunks:
        dataset_id = chunk.get("dataset_id")
        if isinstance(dataset_id, str) and dataset_id.strip():
            ids.append(dataset_id.strip())
    return list(dict.fromkeys(ids))


def collect_responses(
    client: RAGFlowClient,
    chat_id: str,
    samples: list[dict[str, Any]],
    run_id: str,
    raw_path: Path,
    keep_sessions: bool,
    args: argparse.Namespace,
    empty_context_retries: int = 1,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    created_sessions: list[str] = []
    try:
        for idx, sample in enumerate(samples, start=1):
            question = sample["question"]
            raw_row: dict[str, Any] = {
                "sample_id": sample.get("id", f"{run_id}-{idx:04d}"),
                "question": question,
                "source_chunk_id": sample.get("source_chunk_id"),
                "document_name": sample.get("document_name"),
            }
            try:
                answer = ""
                contexts: list[str] = []
                raw_chunks: list[dict[str, Any]] = []
                context_extraction: dict[str, Any] = {}
                retrieval_fallback: dict[str, Any] | None = None
                data: dict[str, Any] = {}
                elapsed = 0.0
                session_id = ""
                retry_count = 0
                for attempt in range(empty_context_retries + 1):
                    session_id = client.create_session(chat_id, f"eval-{run_id}-{idx:04d}-{attempt + 1}")
                    created_sessions.append(session_id)
                    started = time.time()
                    data = client.ask(chat_id, session_id, question, args)
                    answer = data.get("answer") or ""
                    contexts, raw_chunks, context_extraction = contexts_from_response(data)
                    if not contexts and not args.disable_retrieval_fallback:
                        dataset_ids = fallback_dataset_ids(sample, raw_chunks)
                        if dataset_ids:
                            retrieval_fallback = client.retrieve(
                                question,
                                dataset_ids,
                                args,
                                reference_chunk_count=context_extraction.get("reference_chunk_count", 0),
                            )
                            contexts, fallback_chunks, fallback_stats = contexts_from_reference(retrieval_fallback)
                            if contexts:
                                raw_chunks = fallback_chunks
                                context_extraction = {
                                    **context_extraction,
                                    "extraction_source": "retrieval_fallback",
                                    "fallback_dataset_ids": dataset_ids,
                                    "fallback_reference_chunk_count": fallback_stats.get("reference_chunk_count", 0),
                                    "fallback_extracted_context_count": fallback_stats.get("extracted_context_count", 0),
                                    "fallback_field_counts": fallback_stats.get("field_counts", {}),
                                }
                    elapsed = time.time() - started
                    retry_count = attempt
                    if contexts or attempt >= empty_context_retries:
                        break
                    time.sleep(1.5)
                raw_row.update(
                    {
                        "session_id": session_id,
                        "answer": answer,
                        "contexts": contexts,
                        "reference_chunks": raw_chunks,
                        "context_extraction": context_extraction,
                        "retrieval_fallback": retrieval_fallback,
                        "raw_response": data,
                        "elapsed_seconds": elapsed,
                        "retry_count": retry_count,
                        "error": None,
                    }
                )
                records.append(
                    {
                        "sample_id": raw_row["sample_id"],
                        "question": question,
                        "answer": answer,
                        "contexts": contexts,
                        "ground_truth": sample.get("ground_truth", ""),
                        "expected_evidence": sample.get("expected_evidence", ""),
                        "category": sample.get("category", ""),
                        "source_chunk_id": sample.get("source_chunk_id", ""),
                        "document_name": sample.get("document_name", ""),
                        "session_id": session_id,
                        "elapsed_seconds": elapsed,
                        "retry_count": retry_count,
                        "context_extraction": context_extraction,
                    }
                )
                retry_note = f" retries={retry_count}" if retry_count else ""
                print(f"[{idx}/{len(samples)}] contexts={len(contexts)} seconds={elapsed:.2f}{retry_note}")
            except Exception as exc:
                raw_row.update(
                    {
                        "answer": "",
                        "contexts": [],
                        "reference_chunks": [],
                        "context_extraction": {},
                        "raw_response": {},
                        "error": str(exc),
                    }
                )
                print(f"[{idx}/{len(samples)}] ERROR: {exc}", file=sys.stderr)
            append_jsonl(raw_path, raw_row)
    finally:
        if keep_sessions:
            print(f"Keeping {len(created_sessions)} eval sessions for debugging.")
        else:
            try:
                client.delete_sessions(chat_id, created_sessions)
                print(f"Deleted {len(created_sessions)} eval sessions.")
                remaining = client.list_eval_sessions(chat_id, run_id)
                if remaining:
                    print(
                        f"WARNING: {len(remaining)} eval sessions still match eval-{run_id}- after cleanup.",
                        file=sys.stderr,
                    )
            except Exception as exc:
                print(f"WARNING: failed to delete eval sessions: {exc}", file=sys.stderr)
    return records


def records_from_raw_responses(raw_rows: list[dict[str, Any]], samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples_by_id = {str(sample.get("id", "")): sample for sample in samples if sample.get("id")}
    samples_by_question = {str(sample.get("question", "")): sample for sample in samples if sample.get("question")}
    records: list[dict[str, Any]] = []
    for idx, row in enumerate(raw_rows, start=1):
        if row.get("error"):
            continue
        sample_id = str(row.get("sample_id") or f"raw-{idx:04d}")
        sample = samples_by_id.get(sample_id) or samples_by_question.get(str(row.get("question", ""))) or {}
        contexts = row.get("contexts")
        if not isinstance(contexts, list):
            contexts = []
        elapsed = row.get("elapsed_seconds", 0.0)
        try:
            elapsed = float(elapsed)
        except (TypeError, ValueError):
            elapsed = 0.0
        records.append(
            {
                "sample_id": sample_id,
                "question": row.get("question") or sample.get("question", ""),
                "answer": row.get("answer") or "",
                "contexts": contexts,
                "ground_truth": sample.get("ground_truth") or row.get("ground_truth", ""),
                "expected_evidence": sample.get("expected_evidence") or row.get("expected_evidence", ""),
                "category": sample.get("category") or row.get("category", ""),
                "source_chunk_id": sample.get("source_chunk_id") or row.get("source_chunk_id", ""),
                "document_name": sample.get("document_name") or row.get("document_name", ""),
                "session_id": row.get("session_id", ""),
                "elapsed_seconds": elapsed,
                "retry_count": row.get("retry_count", 0),
                "context_extraction": row.get("context_extraction") or {},
            }
        )
    return records


def make_ragas_dataset(records: list[dict[str, Any]]) -> Any:
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'datasets'. Install eval deps with: "
            "uv pip install requests openai ragas datasets langchain-openai pandas"
        ) from exc
    rows = []
    for record in records:
        reference = record.get("ground_truth") or record["expected_evidence"]
        rows.append(
            {
                "user_input": record["question"],
                "response": record["answer"],
                "retrieved_contexts": record["contexts"],
                "reference": reference,
                "question": record["question"],
                "answer": record["answer"],
                "contexts": record["contexts"],
                "ground_truth": reference,
                "sample_id": record["sample_id"],
            }
        )
    return Dataset.from_list(rows)


def build_ragas_metrics() -> list[Any]:
    try:
        import ragas.metrics as metrics_mod
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'ragas'. Install eval deps with: "
            "uv pip install requests openai ragas datasets langchain-openai pandas"
        ) from exc

    selected: list[Any] = []
    for class_name, variable_name in [
        ("Faithfulness", "faithfulness"),
        ("ResponseRelevancy", "answer_relevancy"),
    ]:
        metric = getattr(metrics_mod, class_name, None)
        if metric is not None:
            selected.append(metric())
            continue
        metric = getattr(metrics_mod, variable_name, None)
        if metric is not None:
            selected.append(metric)

    for name in [
        "LLMContextPrecisionWithReference",
        "LLMContextPrecisionWithoutReference",
        "ContextPrecision",
        "context_precision",
    ]:
        metric = getattr(metrics_mod, name, None)
        if metric is not None:
            selected.append(metric() if isinstance(metric, type) else metric)
            break

    for class_name, variable_name in [
        ("AnswerCorrectness", "answer_correctness"),
        ("ContextRecall", "context_recall"),
    ]:
        metric = getattr(metrics_mod, class_name, None)
        if metric is not None:
            selected.append(metric())
            continue
        metric = getattr(metrics_mod, variable_name, None)
        if metric is not None:
            selected.append(metric() if isinstance(metric, type) else metric)

    if not selected:
        raise SystemExit("Could not find compatible Ragas metrics in the installed ragas package.")
    return selected


def is_metric_column(column: str) -> bool:
    return column not in {
        "user_input",
        "response",
        "retrieved_contexts",
        "reference",
        "question",
        "answer",
        "contexts",
        "ground_truth",
        "sample_id",
        "category",
        "context_count",
        "retry_count",
        "context_extraction_source",
    }


def build_ragas_models() -> tuple[Any, Any | None]:
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'langchain-openai'. Install eval deps with: "
            "uv pip install requests openai ragas datasets langchain-openai pandas"
        ) from exc

    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
    except ImportError:
        LangchainEmbeddingsWrapper = None
        LangchainLLMWrapper = None

    base_url = env("DASHSCOPE_BASE_URL")
    if not base_url:
        workspace_id = require_env("DASHSCOPE_WORKSPACE_ID")
        base_url = f"https://{workspace_id}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    api_key = require_env("DASHSCOPE_API_KEY")
    judge_model = env("RAGAS_JUDGE_MODEL", "deepseek-v4-flash")
    embedding_model = env("RAGAS_EMBEDDING_MODEL", "text-embedding-v4")

    chat = ChatOpenAI(
        model=judge_model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        n=1,
        extra_body={"enable_thinking": False},
    )
    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        api_key=api_key,
        base_url=base_url,
        check_embedding_ctx_length=False,
    )
    llm = LangchainLLMWrapper(chat) if LangchainLLMWrapper else chat
    wrapped_embeddings = LangchainEmbeddingsWrapper(embeddings) if LangchainEmbeddingsWrapper else embeddings
    return llm, wrapped_embeddings


def run_ragas(records: list[dict[str, Any]], scores_path: Path) -> dict[str, Any]:
    from ragas import evaluate

    dataset = make_ragas_dataset(records)
    llm, embeddings = build_ragas_models()
    result = evaluate(
        dataset,
        metrics=build_ragas_metrics(),
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
    )
    try:
        df = result.to_pandas()
        for idx, record in enumerate(records):
            if idx >= len(df):
                break
            df.loc[idx, "sample_id"] = record["sample_id"]
            df.loc[idx, "category"] = record.get("category", "")
            df.loc[idx, "context_count"] = len(record["contexts"])
            df.loc[idx, "retry_count"] = record.get("retry_count", 0)
            df.loc[idx, "context_extraction_source"] = record.get("context_extraction", {}).get("extraction_source", "")
        df.to_csv(scores_path, index=False)
        metric_means = {}
        for column in df.columns:
            if not is_metric_column(column):
                continue
            try:
                values = [float(v) for v in df[column].dropna().tolist()]
            except (TypeError, ValueError):
                continue
            if values:
                metric_means[column] = statistics.fmean(values)
    except Exception:
        scores_path.write_text(str(result), encoding="utf-8")
        metric_means = {}
    return {"metric_means": metric_means, "result": str(result)}


def write_fallback_scores(records: list[dict[str, Any]], scores_path: Path) -> None:
    fieldnames = [
        "sample_id",
        "question",
        "answer",
        "context_count",
        "source_chunk_id",
        "document_name",
        "session_id",
        "elapsed_seconds",
        "context_extraction_source",
    ]
    with scores_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "sample_id": record["sample_id"],
                    "question": record["question"],
                    "answer": record["answer"],
                    "context_count": len(record["contexts"]),
                    "source_chunk_id": record["source_chunk_id"],
                    "document_name": record["document_name"],
                    "session_id": record["session_id"],
                    "elapsed_seconds": f"{record['elapsed_seconds']:.3f}",
                    "context_extraction_source": record.get("context_extraction", {}).get("extraction_source", ""),
                }
            )


def build_summary(records: list[dict[str, Any]], ragas_summary: dict[str, Any] | None) -> dict[str, Any]:
    elapsed = [record["elapsed_seconds"] for record in records]
    no_context = [record["sample_id"] for record in records if not record["contexts"]]
    null_content_reference = [
        record["sample_id"]
        for record in records
        if record.get("context_extraction", {}).get("reference_chunk_count", 0) > 0
        and record.get("context_extraction", {}).get("null_content_chunk_count", 0) > 0
    ]
    prompt_fallback = [
        record["sample_id"]
        for record in records
        if record.get("context_extraction", {}).get("extraction_source") == "prompt_fallback"
    ]
    retrieval_fallback = [
        record["sample_id"]
        for record in records
        if record.get("context_extraction", {}).get("extraction_source") == "retrieval_fallback"
    ]
    empty_reference_chunks = [
        record["sample_id"]
        for record in records
        if record.get("context_extraction", {}).get("extraction_source") == "empty_reference_chunks"
    ]
    return {
        "sample_count": len(records),
        "no_context_count": len(no_context),
        "no_context_sample_ids": no_context[:20],
        "null_content_reference_count": len(null_content_reference),
        "null_content_reference_sample_ids": null_content_reference[:20],
        "prompt_context_fallback_count": len(prompt_fallback),
        "prompt_context_fallback_sample_ids": prompt_fallback[:20],
        "retrieval_fallback_count": len(retrieval_fallback),
        "retrieval_fallback_sample_ids": retrieval_fallback[:20],
        "empty_reference_chunks_count": len(empty_reference_chunks),
        "empty_reference_chunks_sample_ids": empty_reference_chunks[:20],
        "latency_seconds_avg": statistics.fmean(elapsed) if elapsed else None,
        "latency_seconds_p50": statistics.median(elapsed) if elapsed else None,
        "ragas": ragas_summary or {},
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=str(Path(__file__).with_name(".env")))
    parser.add_argument("--evalset", required=True)
    parser.add_argument("--run-id", default=time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--output-dir", default=env("RAGAS_EVAL_OUTPUT_DIR", "evals/ragas_ragflow/outputs"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--chat-id")
    parser.add_argument("--dataset-id")
    parser.add_argument("--keep-sessions", action="store_true")
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument("--raw-responses")
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument("--similarity-threshold", type=float, default=0.2)
    parser.add_argument("--vector-similarity-weight", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-n", type=int, default=6)
    parser.add_argument("--rerank-id", default=env("RAGFLOW_RERANK_ID", "qwen3-rerank@agent调用@Tongyi-Qianwen"))
    parser.add_argument("--empty-context-retries", type=int, default=1)
    parser.add_argument("--fallback-retrieval-page-size", type=int, default=30)
    parser.add_argument("--disable-retrieval-fallback", action="store_true")
    parser.add_argument("--disable-table-entity-filter", action="store_true")
    parser.add_argument("--disable-sql-retrieval", action="store_true")
    return parser


def main() -> int:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--env-file", default=str(Path(__file__).with_name(".env")))
    pre_args, _ = pre_parser.parse_known_args()
    load_env_file(pre_args.env_file)

    parser = build_arg_parser()
    args = parser.parse_args()
    load_env_file(args.env_file)

    samples = read_jsonl(Path(args.evalset), limit=args.limit)
    if not samples:
        raise SystemExit("Evalset is empty.")

    output_dir = Path(args.output_dir) / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_responses.jsonl"
    scores_path = output_dir / "scores.csv"
    summary_path = output_dir / "summary.json"

    if args.skip_retrieval:
        source_raw_path = Path(args.raw_responses) if args.raw_responses else raw_path
        if not source_raw_path.exists():
            raise SystemExit(f"Missing raw responses file for --skip-retrieval: {source_raw_path}")
        raw_rows = read_jsonl(source_raw_path)
        records = records_from_raw_responses(raw_rows, samples)
        print(f"Loaded {len(records)} records from {source_raw_path}")
    else:
        client = RAGFlowClient(require_env("RAGFLOW_BASE_URL"), require_env("RAGFLOW_API_KEY"))
        chat_id = args.chat_id
        if not chat_id:
            dataset_id = args.dataset_id or samples[0].get("dataset_id")
            if not dataset_id:
                raise SystemExit("Provide --chat-id or --dataset-id, or include dataset_id in evalset rows.")
            chat = client.create_chat(f"ragas-eval-{args.run_id}", [dataset_id], args)
            chat_id = chat["id"]
            print(f"Created eval chat {chat_id} for dataset {dataset_id}")

        records = collect_responses(
            client,
            chat_id=chat_id,
            samples=samples,
            run_id=args.run_id,
            raw_path=raw_path,
            keep_sessions=args.keep_sessions,
            args=args,
            empty_context_retries=max(args.empty_context_retries, 0),
        )
    if not records:
        raise SystemExit("No successful RAGFlow responses were collected.")

    ragas_summary = None
    if args.skip_ragas:
        write_fallback_scores(records, scores_path)
    else:
        try:
            ragas_summary = run_ragas(records, scores_path)
        except SystemExit as exc:
            message = str(exc)
            print(f"WARNING: Ragas evaluation skipped: {message}", file=sys.stderr)
            write_fallback_scores(records, scores_path)
            ragas_summary = {"error": message}
        except Exception as exc:
            print(f"WARNING: Ragas evaluation failed: {exc}", file=sys.stderr)
            write_fallback_scores(records, scores_path)
            ragas_summary = {"error": str(exc)}

    summary = build_summary(records, ragas_summary)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote raw responses to {raw_path}")
    print(f"Wrote scores to {scores_path}")
    print(f"Wrote summary to {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
