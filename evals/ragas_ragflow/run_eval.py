#!/usr/bin/env python3
"""Run a RAGFlow chat evaluation with Ragas."""

from __future__ import annotations

import argparse
import csv
import json
import os
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
    for line in env_path.read_text(encoding="utf-8").splitlines():
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
        payload = {
            "name": name,
            "dataset_ids": dataset_ids,
            "similarity_threshold": args.similarity_threshold,
            "vector_similarity_weight": args.vector_similarity_weight,
            "top_k": args.top_k,
            "top_n": args.top_n,
            "rerank_id": "",
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

    def ask(self, chat_id: str, session_id: str, question: str) -> dict[str, Any]:
        return self.request(
            "POST",
            "/chat/completions",
            json={"chat_id": chat_id, "session_id": session_id, "question": question, "stream": False},
        )


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


def contexts_from_reference(reference: Any) -> tuple[list[str], list[dict[str, Any]]]:
    if not isinstance(reference, dict):
        return [], []
    contexts: list[str] = []
    raw_chunks: list[dict[str, Any]] = []
    for chunk in reference.get("chunks") or []:
        if not isinstance(chunk, dict):
            continue
        raw_chunks.append(chunk)
        content = chunk.get("content")
        if isinstance(content, str) and content.strip():
            contexts.append(content.strip())
    return contexts, raw_chunks


def collect_responses(
    client: RAGFlowClient,
    chat_id: str,
    samples: list[dict[str, Any]],
    run_id: str,
    raw_path: Path,
    keep_sessions: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    created_sessions: list[str] = []
    try:
        for idx, sample in enumerate(samples, start=1):
            question = sample["question"]
            session_id = client.create_session(chat_id, f"eval-{run_id}-{idx:04d}")
            created_sessions.append(session_id)
            started = time.time()
            raw_row: dict[str, Any] = {
                "sample_id": sample.get("id", f"{run_id}-{idx:04d}"),
                "session_id": session_id,
                "question": question,
                "source_chunk_id": sample.get("source_chunk_id"),
                "document_name": sample.get("document_name"),
            }
            try:
                data = client.ask(chat_id, session_id, question)
                answer = data.get("answer") or ""
                contexts, raw_chunks = contexts_from_reference(data.get("reference"))
                elapsed = time.time() - started
                raw_row.update(
                    {
                        "answer": answer,
                        "contexts": contexts,
                        "reference_chunks": raw_chunks,
                        "elapsed_seconds": elapsed,
                        "error": None,
                    }
                )
                records.append(
                    {
                        "sample_id": raw_row["sample_id"],
                        "question": question,
                        "answer": answer,
                        "contexts": contexts,
                        "expected_evidence": sample.get("expected_evidence", ""),
                        "source_chunk_id": sample.get("source_chunk_id", ""),
                        "document_name": sample.get("document_name", ""),
                        "session_id": session_id,
                        "elapsed_seconds": elapsed,
                    }
                )
                print(f"[{idx}/{len(samples)}] contexts={len(contexts)} seconds={elapsed:.2f}")
            except Exception as exc:
                raw_row.update({"answer": "", "contexts": [], "reference_chunks": [], "error": str(exc)})
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
        rows.append(
            {
                "user_input": record["question"],
                "response": record["answer"],
                "retrieved_contexts": record["contexts"],
                "reference": record["expected_evidence"],
                "question": record["question"],
                "answer": record["answer"],
                "contexts": record["contexts"],
                "ground_truth": record["expected_evidence"],
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

    if not selected:
        raise SystemExit("Could not find compatible Ragas metrics in the installed ragas package.")
    return selected


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

    chat = ChatOpenAI(model=judge_model, api_key=api_key, base_url=base_url, temperature=0)
    embeddings = OpenAIEmbeddings(model=embedding_model, api_key=api_key, base_url=base_url)
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
        df.to_csv(scores_path, index=False)
        metric_means = {}
        for column in df.columns:
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
                }
            )


def build_summary(records: list[dict[str, Any]], ragas_summary: dict[str, Any] | None) -> dict[str, Any]:
    elapsed = [record["elapsed_seconds"] for record in records]
    no_context = [record["sample_id"] for record in records if not record["contexts"]]
    return {
        "sample_count": len(records),
        "no_context_count": len(no_context),
        "no_context_sample_ids": no_context[:20],
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
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument("--similarity-threshold", type=float, default=0.2)
    parser.add_argument("--vector-similarity-weight", type=float, default=0.3)
    parser.add_argument("--top-k", type=int, default=1024)
    parser.add_argument("--top-n", type=int, default=6)
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
