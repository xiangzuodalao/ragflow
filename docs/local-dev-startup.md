# Local RAGFlow Startup Guide

This document records the working local startup flow for this workspace:

- Repo path on Windows: `D:\code\ragflow`
- Repo path in WSL: `/mnt/d/code/ragflow`
- Backend runs in WSL `Ubuntu-24.04`
- Dependency services run separately through Docker/Docker Desktop
- Frontend runs on Windows from `web/`

Use this when the local RAGFlow backend or frontend has been stopped and needs
to be brought back up.

## 1. Start Dependency Services

RAGFlow backend expects MySQL, Redis, MinIO, Elasticsearch and related services
to be reachable on localhost ports from WSL.

If Docker is available in WSL:

```bash
cd /mnt/d/code/ragflow
docker compose -f docker/docker-compose-base.yml up -d
docker compose -f docker/docker-compose-base.yml ps
```

If WSL says `docker: command not found`, Docker Desktop WSL integration is not
enabled for `Ubuntu-24.04`. Fix it from Docker Desktop:

```text
Docker Desktop -> Settings -> Resources -> WSL Integration -> enable Ubuntu-24.04
```

Then open a fresh WSL shell and rerun the compose commands.

Quick dependency checks from WSL:

```bash
curl -sS -m 5 http://127.0.0.1:1200/_cluster/health
nc -zv 127.0.0.1 3307
nc -zv 127.0.0.1 6379
nc -zv 127.0.0.1 9000
```

## 2. Start Backend

Run this from Windows PowerShell:

```powershell
wsl -d Ubuntu-24.04 -e bash -lc 'cd /mnt/d/code/ragflow && mkdir -p logs && setsid env PYTHONPATH=/mnt/d/code/ragflow NLTK_DATA=/mnt/d/code/ragflow/nltk_data .venv/bin/python api/ragflow_server.py >> logs/ragflow_server.log 2>&1 < /dev/null & disown; sleep 1; pgrep -af ragflow_server.py || true'
```

Important details:

- Use `.venv/bin/python`, not the raw uv Python path. The raw interpreter does
  not include project dependencies such as `quart`.
- Use `setsid ... & disown` so the backend survives after the `wsl.exe` command
  returns.
- Do not run `api/ragflow_server.py` directly as a shell command. If you see
  `syntax error near unexpected token "Start RAGFlow server..."`, the Python
  file was executed by shell instead of Python.

## 3. Verify Backend

From WSL:

```bash
curl -sS -m 5 http://127.0.0.1:9380/api/v1/system/healthz
```

Expected:

```json
{"db":"ok","doc_engine":"ok","redis":"ok","status":"ok","storage":"ok"}
```

From Windows PowerShell:

```powershell
Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 http://127.0.0.1:9380/api/v1/system/healthz
```

Check process and port from WSL:

```bash
pgrep -af ragflow_server.py
ss -ltnp | grep 9380
```

Tail backend logs:

```bash
cd /mnt/d/code/ragflow
tail -120 logs/ragflow_server.log
```

If health check is not ready immediately, wait. A cold backend startup can take
around 2 to 3 minutes.

## 4. Start Frontend

Run this from Windows PowerShell:

```powershell
cd D:\code\ragflow\web
npm run dev
```

If startup fails with `Error: spawn EPERM` while loading Vite/esbuild, rerun the
same command with elevated permission. In Codex, that means rerunning
`npm run dev` with escalation.

Vite may choose a different port if the default is occupied. In the most recent
run:

```text
Port 9222 is in use, trying another one...
Local: http://localhost:9223/
```

Use the URL printed by Vite.

## 5. Verify Frontend

From Windows PowerShell:

```powershell
Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 http://localhost:9223/
```

Expected status: `200`.

If Vite printed another port, replace `9223` with that port.

## 6. Stop Services

Stop frontend:

```text
Press Ctrl+C in the terminal running npm run dev.
```

Stop backend from WSL:

```bash
pkill -f 'api/ragflow_server.py'
```

Confirm backend stopped:

```bash
pgrep -af ragflow_server.py || true
curl -sS -m 2 http://127.0.0.1:9380/api/v1/system/healthz
```

Stop dependency services only when you actually want to stop them:

```bash
cd /mnt/d/code/ragflow
docker compose -f docker/docker-compose-base.yml down
```

## 7. Common Failures

### Backend health check connection refused

Check whether backend is running:

```bash
pgrep -af ragflow_server.py
ss -ltnp | grep 9380
tail -120 /mnt/d/code/ragflow/logs/ragflow_server.log
```

If there is no process, start backend again with the command in section 2.

### `docker` not found in WSL

Docker Desktop WSL integration is missing or disabled for `Ubuntu-24.04`.
Enable it in Docker Desktop settings, then reopen WSL.

### `ModuleNotFoundError: No module named 'quart'`

The backend was started with the wrong Python interpreter. Use:

```bash
.venv/bin/python api/ragflow_server.py
```

Do not use:

```bash
/root/.local/share/uv/python/.../python3.13 api/ragflow_server.py
```

### Shell syntax error near `Start RAGFlow server...`

The Python file was executed as a shell script. Start it through Python:

```bash
.venv/bin/python api/ragflow_server.py
```

### Frontend `spawn EPERM`

Vite needs to execute the local esbuild binary. Rerun `npm run dev` with
elevated permission.

## 8. Known Good Commands

Backend:

```powershell
wsl -d Ubuntu-24.04 -e bash -lc 'cd /mnt/d/code/ragflow && mkdir -p logs && setsid env PYTHONPATH=/mnt/d/code/ragflow NLTK_DATA=/mnt/d/code/ragflow/nltk_data .venv/bin/python api/ragflow_server.py >> logs/ragflow_server.log 2>&1 < /dev/null & disown; sleep 1; pgrep -af ragflow_server.py || true'
```

Backend health:

```powershell
Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 http://127.0.0.1:9380/api/v1/system/healthz
```

Frontend:

```powershell
cd D:\code\ragflow\web
npm run dev
```
