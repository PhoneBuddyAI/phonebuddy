# AndroidWorld 116-Task Benchmark Runner (Local Deployment + vLLM Agent)

> 中文版见 [`README_zh.md`](./README_zh.md).

This document records the full process of deploying the AndroidWorld service from
scratch on **macOS (Apple Silicon / arm64)**, wiring up a vLLM-served model as the
GUI Agent, and running the entire AndroidWorld 116-task benchmark end to end.

Companion script: `run_android_world.py` (standalone, **no external agent
framework dependency** — the HTTP client, the Qwen3.5-SFT-EN prompt, response
parsing, action mapping, and the vLLM client are all implemented inline).

---

## 0. Architecture Overview

```
┌─────────────────────┐   HTTP (OpenAI-compatible) ┌──────────────────────┐
│ run_android_world.py │ ─────────────────────────► │  vLLM service         │
│   (agent loop)       │  /v1/chat/completions      │  YOUR_VLLM_HOST:PORT  │
│                      │ ◄───────────────────────── │  (Qwen3.5 toolcall)   │
│                      │   <tool_call>{...}         └──────────────────────┘
│                      │
│                      │   HTTP                     ┌──────────────────────┐
│                      │ ─────────────────────────► │  AndroidWorld server  │
│                      │  /screenshot /reset        │  127.0.0.1:5001       │
│                      │  /execute_action ...       │  (server.android_     │
│                      │ ◄───────────────────────── │   server, FastAPI)    │
└─────────────────────┘                             └───────────┬──────────┘
                                                                 │ gRPC 8554 / adb
                                                                 ▼
                                                    ┌──────────────────────┐
                                                    │  Android emulator     │
                                                    │  AndroidWorldAvd      │
                                                    └──────────────────────┘
```

Per-task flow: `reset → reinitialize_suite → initialize_task → [screenshot → model → parse → execute]×N → score → tear_down`

---

## 1. Prerequisites

- macOS (this guide uses Apple Silicon / arm64 as the example)
- **conda / Python 3.11+** (verified working on conda base Python 3.13, with a few
  extra patches — see below)
- Android SDK (including `emulator` and `platform-tools/adb`); path used here:
  `~/Library/Android/sdk`
- A pre-created AVD: `AndroidWorldAvd` (Pixel 6 / API 33, created per the official
  AndroidWorld README)
- A reachable vLLM OpenAI-compatible service (referred to below as
  `http://YOUR_VLLM_HOST:PORT` — replace with your own address)

---

## 2. Environment Setup

> ⚠️ The steps below are the complete, battle-tested procedure for a **Python 3.13 +
> conda** environment. If you use Python 3.11, the `audioop` / `imp` pitfalls may not
> apply, but proto generation and the SQLite FTS4 fix are still needed.

### 2.1 Install Python dependencies

```bash
# Base dependencies
pip install -r requirements.txt          # absl-py, android_env, numpy, opencv-python, ...
pip install fastapi uvicorn pydantic      # needed by the server (in pyproject.toml, not requirements.txt)
pip install openai                        # for calling vLLM

# Python 3.13-specific patches:
pip install audioop-lts                   # pydub depends on audioop, removed in 3.13 — this is the backport
pip install IPython matplotlib pandas pytest termcolor   # not fully covered by requirements

# Needed to generate proto:
pip install "grpcio-tools==1.71.0"
```

### 2.2 Generate protobuf bindings (optional)

> Only needed if the server fails to start with an import error like `state_pb2` /
> `task_pb2`. If the repo already ships precompiled `*_pb2.py`, you can skip this.

If the `*.proto` files are not precompiled, `setup.py` will also fail on Python 3.13
due to `import imp`. In that case, **bypass setup.py and compile directly with
grpc_tools**:

```bash
cd <repo>/android_world      # i.e. this directory

ROOT="$(pwd)"
INCLUDE="$(python3 -c 'import os,grpc_tools; print(os.path.join(os.path.dirname(grpc_tools.__file__), "_proto"))')"

for p in \
  android_world/task_evals/information_retrieval/proto/state.proto \
  android_world/task_evals/information_retrieval/proto/task.proto ; do
  python3 -m grpc_tools.protoc \
    --proto_path="$INCLUDE" --proto_path="$ROOT" \
    --python_out="$ROOT" --grpc_python_out="$ROOT" \
    "$ROOT/$p"
done

# Verify:
python3 -c "from android_world.task_evals.information_retrieval.proto import state_pb2, task_pb2; print('proto OK')"
```

### 2.3 Fix SQLite FTS4 (required)

When AndroidWorld sets up the **Joplin** app, it uses Python's `sqlite3` to operate on
an **FTS4 virtual table**. However, the `libsqlite3` shipped with conda is often
compiled with FTS5 only — no FTS3/FTS4 — so the server crashes on Joplin at startup.

**Solution: compile your own `libsqlite3.dylib` with FTS3/4/5 from the official
amalgamation** (conda-forge's libsqlite pulls in ICU and triggers a dependency
cascade; pysqlite3-binary has no macOS wheel — hence self-compiling):

```bash
# 1) Download the SQLite amalgamation (version per the official download.html; mind the year directory)
cd /tmp
curl -fsSL -o sqlite-amalg.zip "https://www.sqlite.org/2026/sqlite-amalgamation-3530200.zip"
unzip -o sqlite-amalg.zip

# 2) Compile to a dylib with FTS3/4/5 enabled; set compatibility version to 9.0.0 to match _sqlite3.so
mkdir -p ~/.local/lib/androidworld-sqlite
clang -O2 -dynamiclib -fPIC \
  -DSQLITE_ENABLE_FTS3 -DSQLITE_ENABLE_FTS3_PARENTHESIS -DSQLITE_ENABLE_FTS4 -DSQLITE_ENABLE_FTS5 \
  -DSQLITE_ENABLE_RTREE -DSQLITE_ENABLE_JSON1 -DSQLITE_ENABLE_COLUMN_METADATA \
  -DSQLITE_ENABLE_DBSTAT_VTAB -DSQLITE_THREADSAFE=1 \
  -install_name @rpath/libsqlite3.dylib \
  -compatibility_version 9.0.0 -current_version 9.6.0 \
  sqlite-amalgamation-3530200/sqlite3.c \
  -o ~/.local/lib/androidworld-sqlite/libsqlite3.dylib

# 3) Verify FTS4 works
DYLD_LIBRARY_PATH="$HOME/.local/lib/androidworld-sqlite" python3 -c \
  "import sqlite3; c=sqlite3.connect(':memory:'); c.execute('CREATE VIRTUAL TABLE t USING fts4(x)'); print('FTS4 OK', sqlite3.sqlite_version)"
```

> Afterwards, **you must start the server with `DYLD_LIBRARY_PATH="$HOME/.local/lib/androidworld-sqlite"`**
> so Python loads this FTS4-enabled sqlite.

---

## 3. Start the emulator

> Before first use, follow **step 1 of the Installation section** in the
> [google-research/android_world](https://github.com/google-research/android_world)
> README.md to set up the Android Emulator (create `AndroidWorldAvd`), then run the
> launch command below.

`emulator` is a foreground/blocking command that holds the current terminal. Start it
**in one terminal**:

```bash
cd ~/Library/Android/sdk/emulator
./emulator -avd AndroidWorldAvd -no-snapshot -grpc 8554
```

**In a separate terminal**, wait for the system to fully boot (you must wait for
`boot_completed=1` before starting the server, otherwise emulator_setup will fail):

```bash
~/Library/Android/sdk/platform-tools/adb wait-for-device
~/Library/Android/sdk/platform-tools/adb -s emulator-5554 shell getprop sys.boot_completed   # returns 1 when boot is complete
```

The emulator should be listening on: console `5554`, adb `5555`, gRPC `8554`.

---

## 4. Start the AndroidWorld server

> **Before starting, you must edit `server/android_server.py`** (in `lifespan()` the
> adb path is hardcoded to a Docker path, and the port 5000 is occupied by macOS
> AirPlay):
> ```python
> adb_path="$HOME/Library/Android/sdk/platform-tools/adb",   # set to the real adb path on your machine; was /opt/android/platform-tools/adb
> uvicorn.run(app, host="127.0.0.1", port=5001)              # end of file, was port=5000
> ```

```bash
cd <repo>/android_world

DYLD_LIBRARY_PATH="$HOME/.local/lib/androidworld-sqlite" python3 -m server.android_server
```

On startup it runs `emulator_setup`, which **installs/configures 24 apps and takes
about 5–10 minutes**.

- You will see many `WARNING:absl:Failed to automatically setup app xxx ... Target text ... not found`
  — **non-fatal**; it just couldn't auto-dismiss a permission dialog and continues.
- The `VLC` install reports `INSTALL_FAILED_NO_MATCHING_ABIS` — **known issue** (the
  VLC apk is x86, incompatible with the arm64 emulator); only affects the 2 Vlc tasks,
  everything else is fine.
- Once you see `Application startup complete.` + `Uvicorn running on http://127.0.0.1:5001`, it's ready.

Health check:

```bash
curl -s http://127.0.0.1:5001/health                               # {"status":"success"}
curl -s "http://127.0.0.1:5001/suite/task_list?max_index=10000" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['task_list']))"   # 116
curl -s -X POST "http://127.0.0.1:5001/reset?go_home=true"         # {"status":"success",...}
```

---

## 5. Run the benchmark

### 5.1 All 116 tasks

> The test script `run_android_world.py` lives alongside the `android_world/`
> directory (both under this `eval/` directory). It talks to the server over HTTP, so
> it does **not** need to run inside the `android_world/` package directory.
> The script itself does not need the FTS4 sqlite, but passing `DYLD_LIBRARY_PATH` is
> harmless (only the server truly needs it).

```bash
cd <repo>            # the script's directory (the parent of android_world/)

python3 run_android_world.py \
    --max-steps 50 --step-delay 2 --seed 30 --resume \
    --output-dir ./aw116_output
```

Runs serially (single device); the full run takes a few hours.

### 5.2 Common arguments

| Argument | Default | Description |
|---|---|---|
| `--aw-server`     | `http://127.0.0.1:5001`         | AndroidWorld server address |
| `--vllm-base-url` | `http://YOUR_VLLM_HOST:PORT/v1` | vLLM OpenAI-compatible address |
| `--vllm-model`    | auto-discover                   | leave empty to take the first from `/v1/models` |
| `--max-steps`     | 50                              | max steps per task |
| `--step-delay`    | 2                               | delay between steps (seconds) |
| `--seed`          | 30                              | suite random seed |
| `--limit N`       | none                            | only run the first N tasks (for smoke tests) |
| `--tasks A,B,C`   | none                            | only run the specified task_types |
| `--resume`        | off                             | skip task_types already present in results.jsonl |
| `--save-images`   | off                             | save a screenshot for each step |

You can also override via environment variables: `AW_SERVER_URL` / `VLLM_BASE_URL` /
`VLLM_API_KEY` / `VLLM_MODEL`.

### 5.3 Smoke test (verify the pipeline first)

```bash
cd <repo>            # the script's directory (the parent of android_world/)

python3 run_android_world.py --tasks ContactsAddContact --max-steps 8 --output-dir ./aw116_smoke
```

---

## 6. Output files (under `--output-dir`)

- `results.jsonl` — one line per task: `task_type, goal, score, success, steps, terminated_by, error, trajectory (per-step actions)`
- `summary.json` — overall success rate + per-task details (written when the script finishes)
- `images/<task>/...` (if `--save-images` is on)

Check live progress / stats:

```bash
python3 - <<'PY'
import json
recs=[json.loads(l) for l in open("./aw116_output/results.jsonl") if l.strip()]
n=len(recs); s=sum(1 for r in recs if r.get("success"))
print(f"done {n}/116 | success {s} ({s/n*100:.1f}%) | errors {sum(1 for r in recs if r.get('error'))}")
PY
```

---

## 7. Key files

> Paths are relative to this `eval/` directory. The test script and this document are
> under `eval/`, alongside `android_world/`.

| File | Purpose |
|---|---|
| `run_android_world.py`            | Standalone test script (agent loop + vLLM client + action mapping) |
| `README.md`                           | This document |
| `android_world/server/android_server.py`         | AndroidWorld FastAPI server (needs adb path + port 5001 edits, see section 4) |
| `~/.local/lib/androidworld-sqlite/libsqlite3.dylib` | Self-compiled FTS4-enabled sqlite, loaded via `DYLD_LIBRARY_PATH` when starting the server |
