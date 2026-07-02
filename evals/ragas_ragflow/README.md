# RAGFlow Ragas Evaluation

This directory contains a lightweight external evaluation harness for RAGFlow.
It samples real chunks from a RAGFlow dataset, generates grounded questions, runs
each question through a RAGFlow chat, and scores the answers with Ragas.

No secrets should be committed. Put local credentials in
`evals/ragas_ragflow/.env` or export them in your shell.

## Setup

Install the evaluation-only dependencies:

```bash
uv pip install requests openai ragas datasets langchain-openai pandas
```

On Windows, prefer running the Ragas scoring step in WSL/Python 3.12 if the
local Python/NumPy stack is unstable. The combination below was used for the
initial Bailian baseline:

```bash
python3 -m venv /tmp/ragas_eval_venv
/tmp/ragas_eval_venv/bin/python -m pip install \
  requests openai ragas datasets langchain-openai pandas
/tmp/ragas_eval_venv/bin/python -m pip install \
  "langchain-community==0.2.19" \
  "langchain-core==0.2.43" \
  "langchain-openai==0.1.25" \
  "numpy<2"
```

Create a local env file from `config.example.env`:

```bash
cp evals/ragas_ragflow/config.example.env evals/ragas_ragflow/.env
```

Required variables:

- `RAGFLOW_BASE_URL`
- `RAGFLOW_API_KEY`
- `DASHSCOPE_WORKSPACE_ID`
- `DASHSCOPE_API_KEY`

The default judge endpoint is:

```text
https://${DASHSCOPE_WORKSPACE_ID}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
```

The default judge model is `deepseek-v4-flash`.

## Generate An Eval Set

Generate 50 questions from the largest parsed dataset available to the API key:

```bash
python evals/ragas_ragflow/generate_evalset.py
```

Generate a smaller smoke set:

```bash
python evals/ragas_ragflow/generate_evalset.py --sample-size 5 --run-id smoke
```

Use a specific dataset:

```bash
python evals/ragas_ragflow/generate_evalset.py --dataset-id <DATASET_ID>
```

Output:

```text
evals/ragas_ragflow/outputs/{run_id}/evalset.jsonl
```

Each row contains `question`, `expected_evidence`, source chunk metadata,
`category`, and `difficulty`.

## Run Evaluation

Run a smoke evaluation and skip Ragas scoring:

```bash
python evals/ragas_ragflow/run_eval.py \
  --evalset evals/ragas_ragflow/outputs/smoke/evalset.jsonl \
  --limit 1 \
  --skip-ragas
```

Run a 5-sample check with Ragas:

```bash
python evals/ragas_ragflow/run_eval.py \
  --evalset evals/ragas_ragflow/outputs/smoke/evalset.jsonl \
  --limit 5
```

Run a full baseline:

```bash
python evals/ragas_ragflow/run_eval.py \
  --evalset evals/ragas_ragflow/outputs/{run_id}/evalset.jsonl
```

If `--chat-id` is not supplied, `run_eval.py` creates a chat named
`ragas-eval-{run_id}` using the dataset ID from the eval set.

## Session Isolation

Each question creates one independent session:

1. `POST /api/v1/chats/{chat_id}/sessions`
2. `POST /api/v1/chat/completions` with `chat_id`, `session_id`, and
   `stream=false`
3. `DELETE /api/v1/chats/{chat_id}/sessions` with the created session IDs

Use `--keep-sessions` when debugging. By default, eval sessions are deleted at
the end of the run. If a response has no contexts, the runner retries once with
a new independent session by default. Use `--empty-context-retries 0` to disable
that behavior.

## Outputs

Each run writes:

```text
evals/ragas_ragflow/outputs/{run_id}/raw_responses.jsonl
evals/ragas_ragflow/outputs/{run_id}/scores.csv
evals/ragas_ragflow/outputs/{run_id}/summary.json
```

`raw_responses.jsonl` keeps the complete raw response and reference chunk
payload. Ragas receives `contexts` as `list[str]`, extracted from
`reference.chunks[*].content`.

## Default Evaluation Settings

When `run_eval.py` creates a chat, it uses:

- `similarity_threshold=0.2`
- `vector_similarity_weight=0.3`
- `top_k=1024`
- `top_n=6`
- no rerank
- no knowledge graph
- no cross-language search

The initial Ragas metrics are faithfulness, answer relevancy, and context
precision where supported by the installed Ragas version. The Bailian
`deepseek-v4-flash` judge is called with `n=1` and `enable_thinking=false` for
OpenAI-compatible API compatibility. Bailian `text-embedding-v4` is called
through LangChain with local embedding context splitting disabled.

After adding manually reviewed `ground_truth` answers, extend the run with
answer correctness and context recall.
