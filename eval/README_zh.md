# AndroidWorld 116-Task Benchmark Runner (本地部署 + vLLM Agent)

> English version: [`README.md`](./README.md).

本文档记录在 **macOS (Apple Silicon / arm64)** 上从零部署 AndroidWorld 服务、
对接 vLLM 模型作为 GUI Agent、并完整跑通 AndroidWorld 116 道题的全过程。

配套脚本：`run_android_world.py`（独立脚本，**无外部 agent 框架依赖**，
所有逻辑——HTTP 客户端、Qwen3.5-SFT-EN prompt、响应解析、动作映射、vLLM 客户端——均内联实现）。

---

## 0. 架构总览

```
┌─────────────────────┐   HTTP (OpenAI 兼容)   ┌──────────────────────┐
│ run_android_world.py │ ─────────────────────► │  vLLM 服务            │
│   (Agent 主循环)      │  /v1/chat/completions  │  YOUR_VLLM_HOST:PORT  │
│                      │ ◄───────────────────── │  (Qwen3.5 toolcall)   │
│                      │   <tool_call>{...}     └──────────────────────┘
│                      │
│                      │   HTTP                 ┌──────────────────────┐
│                      │ ─────────────────────► │  AndroidWorld server  │
│                      │  /screenshot /reset    │  127.0.0.1:5001       │
│                      │  /execute_action ...   │  (server.android_     │
│                      │ ◄───────────────────── │   server, FastAPI)    │
└─────────────────────┘                         └───────────┬──────────┘
                                                             │ gRPC 8554 / adb
                                                             ▼
                                                ┌──────────────────────┐
                                                │  Android 模拟器        │
                                                │  AndroidWorldAvd      │
                                                └──────────────────────┘
```

每道题的流程：`reset → reinitialize_suite → initialize_task → [截图→模型→解析→执行]×N → score → tear_down`

---

## 1. 前置要求

- macOS（本文以 Apple Silicon / arm64 为例）
- **conda / Python 3.11+**（实测在 conda base 的 Python 3.13 上跑通，需额外打几个补丁，见下文）
- Android SDK（含 `emulator` 和 `platform-tools/adb`），本文路径：
  `~/Library/Android/sdk`
- 一个已创建的 AVD：`AndroidWorldAvd`（Pixel 6 / API 33，按 AndroidWorld 官方 README 创建）
- 一个可访问的 vLLM OpenAI 兼容服务（下文用 `http://YOUR_VLLM_HOST:PORT` 表示，替换成你自己的地址）

---

## 2. 环境安装

> ⚠️ 以下是 **Python 3.13 + conda** 环境实际踩坑后的完整步骤。如果你用 Python 3.11，
> `audioop`、`imp` 那两个坑可能不存在，但 proto 生成和 SQLite FTS4 仍需处理。

### 2.1 安装 Python 依赖

```bash
# 基础依赖
pip install -r requirements.txt          # absl-py, android_env, numpy, opencv-python, ...
pip install fastapi uvicorn pydantic      # server 用（pyproject.toml 里有，requirements.txt 没有）
pip install openai                        # 调 vLLM 用

# Python 3.13 专属补丁：
pip install audioop-lts                   # pydub 依赖的 audioop 在 3.13 被移除，需此 backport
pip install IPython matplotlib pandas pytest termcolor   # requirements 里没装全的

# 生成 proto 需要：
pip install "grpcio-tools==1.71.0"
```

### 2.2 生成 protobuf 绑定（可选）

> 仅当 server 启动时报 `state_pb2` / `task_pb2` 之类的 import 错误才需要执行；
> 若仓库里已带有预编译的 `*_pb2.py` 则可跳过。

仓库里 `*.proto` 若没有预编译，`setup.py` 在 Python 3.13 上又会因 `import imp` 失败，
此时**绕过 setup.py 直接用 grpc_tools 编译**：

```bash
cd <repo>/android_world      # 即本目录

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

# 验证：
python3 -c "from android_world.task_evals.information_retrieval.proto import state_pb2, task_pb2; print('proto OK')"
```

### 2.3 修复 SQLite FTS4（必须）

AndroidWorld 的 **Joplin** app 配置时会用 Python 的 `sqlite3` 操作一个 **FTS4 虚拟表**，
而 conda 自带的 `libsqlite3` 往往只编译了 FTS5、没有 FTS3/FTS4 → server 启动时崩在 Joplin。

**解决方案：自己从官方 amalgamation 编译一个带 FTS3/4/5 的 `libsqlite3.dylib`**
（conda-forge 的 libsqlite 依赖 ICU 会引发依赖级联，pysqlite3-binary 无 macOS wheel，故选自编译）：

```bash
# 1) 下载 SQLite amalgamation（版本号按官网 download.html，注意年份目录）
cd /tmp
curl -fsSL -o sqlite-amalg.zip "https://www.sqlite.org/2026/sqlite-amalgamation-3530200.zip"
unzip -o sqlite-amalg.zip

# 2) 编译为 dylib，启用 FTS3/4/5，compatibility version 设为 9.0.0 以匹配 _sqlite3.so
mkdir -p ~/.local/lib/androidworld-sqlite
clang -O2 -dynamiclib -fPIC \
  -DSQLITE_ENABLE_FTS3 -DSQLITE_ENABLE_FTS3_PARENTHESIS -DSQLITE_ENABLE_FTS4 -DSQLITE_ENABLE_FTS5 \
  -DSQLITE_ENABLE_RTREE -DSQLITE_ENABLE_JSON1 -DSQLITE_ENABLE_COLUMN_METADATA \
  -DSQLITE_ENABLE_DBSTAT_VTAB -DSQLITE_THREADSAFE=1 \
  -install_name @rpath/libsqlite3.dylib \
  -compatibility_version 9.0.0 -current_version 9.6.0 \
  sqlite-amalgamation-3530200/sqlite3.c \
  -o ~/.local/lib/androidworld-sqlite/libsqlite3.dylib

# 3) 验证 FTS4 可用
DYLD_LIBRARY_PATH="$HOME/.local/lib/androidworld-sqlite" python3 -c \
  "import sqlite3; c=sqlite3.connect(':memory:'); c.execute('CREATE VIRTUAL TABLE t USING fts4(x)'); print('FTS4 OK', sqlite3.sqlite_version)"
```

> 之后**启动 server 时必须带上 `DYLD_LIBRARY_PATH="$HOME/.local/lib/androidworld-sqlite"`**，
> 让 Python 加载这个带 FTS4 的 sqlite。

---

## 3. 启动模拟器

> 首次使用前，请先参考 [google-research/android_world](https://github.com/google-research/android_world) 的 README.md 中 **Installation 的第 1 步**配置 Android Emulator（创建 `AndroidWorldAvd`），之后再执行下面的启动命令。

`emulator` 是前台阻塞命令，启动后会占住当前终端。**在一个终端**启动它：

```bash
cd ~/Library/Android/sdk/emulator
./emulator -avd AndroidWorldAvd -no-snapshot -grpc 8554
```

**另开一个终端**等待系统真正开机完成（`boot_completed=1` 后才能启动 server，
否则 emulator_setup 会失败）：

```bash
~/Library/Android/sdk/platform-tools/adb wait-for-device
~/Library/Android/sdk/platform-tools/adb -s emulator-5554 shell getprop sys.boot_completed   # 返回 1 即开机完成
```

模拟器需监听：console `5554`、adb `5555`、gRPC `8554`。

---

## 4. 启动 AndroidWorld server

> **启动前必改 `server/android_server.py`**（`lifespan()` 里 adb 路径硬编码成 Docker 路径、
> 端口是被 macOS AirPlay 占用的 5000）：
> ```python
> adb_path="$HOME/Library/Android/sdk/platform-tools/adb",   # 改成你机器上 adb 的实际路径；原 /opt/android/platform-tools/adb
> uvicorn.run(app, host="127.0.0.1", port=5001)              # 文件末尾，原 port=5000
> ```

```bash
cd <repo>/android_world

DYLD_LIBRARY_PATH="$HOME/.local/lib/androidworld-sqlite" python3 -m server.android_server
```

启动时会跑 `emulator_setup`，**安装/配置 24 个 app，耗时约 5~10 分钟**。

- 期间会有大量 `WARNING:absl:Failed to automatically setup app xxx ... Target text ... not found` —— **非致命**，
  只是没能自动点掉某个权限弹窗，会继续。
- `VLC` 安装会报 `INSTALL_FAILED_NO_MATCHING_ABIS` —— **已知问题**（VLC apk 是 x86 的，
  与 arm64 模拟器不匹配），只影响 2 道 Vlc 题，其余正常。
- 看到 `Application startup complete.` + `Uvicorn running on http://127.0.0.1:5001` 即就绪。

健康检查：

```bash
curl -s http://127.0.0.1:5001/health                               # {"status":"success"}
curl -s "http://127.0.0.1:5001/suite/task_list?max_index=10000" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['task_list']))"   # 116
curl -s -X POST "http://127.0.0.1:5001/reset?go_home=true"         # {"status":"success",...}
```

---

## 5. 跑测试

### 5.1 全量 116 题

> 测试脚本 `run_android_world.py` 与 `android_world/` 目录同级（同在本 `eval/` 目录下），
> 通过 HTTP 访问 server，**无需**在 `android_world/` 包目录内运行。
> 脚本自身不需要 FTS4 的 sqlite，但带上 `DYLD_LIBRARY_PATH` 也无妨（仅 server 真正需要它）。

```bash
cd <repo>            # 脚本所在目录（android_world/ 的上一级）

python3 run_android_world.py \
    --max-steps 50 --step-delay 2 --seed 30 --resume \
    --output-dir ./aw116_output
```

串行执行（单设备），全程约数小时。

### 5.2 常用参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--aw-server`     | `http://127.0.0.1:5001`        | AndroidWorld server 地址 |
| `--vllm-base-url` | `http://YOUR_VLLM_HOST:PORT/v1` | vLLM OpenAI 兼容地址 |
| `--vllm-model`    | 自动发现                        | 留空则从 `/v1/models` 取第一个 |
| `--max-steps`     | 50                             | 每题最大步数 |
| `--step-delay`    | 2                              | 步间延迟（秒） |
| `--seed`          | 30                             | suite 随机种子 |
| `--limit N`       | 无                             | 只跑前 N 题（冒烟测试用） |
| `--tasks A,B,C`   | 无                             | 只跑指定 task_type |
| `--resume`        | 关                             | 跳过 results.jsonl 中已出现的 task_type |
| `--save-images`   | 关                             | 保存每步截图 |

也可用环境变量覆盖：`AW_SERVER_URL` / `VLLM_BASE_URL` / `VLLM_API_KEY` / `VLLM_MODEL`。

### 5.3 冒烟测试（先验证链路）

```bash
cd <repo>            # 脚本所在目录（android_world/ 的上一级）

python3 run_android_world.py --tasks ContactsAddContact --max-steps 8 --output-dir ./aw116_smoke
```

---

## 6. 输出文件（`--output-dir` 下）

- `results.jsonl` —— 每题一行：`task_type, goal, score, success, steps, terminated_by, error, trajectory(每步动作)`
- `summary.json` —— 总成功率 + 每题明细（脚本结束时写）
- `images/<task>/...`（若开 `--save-images`）

查看实时进度 / 统计：

```bash
python3 - <<'PY'
import json
recs=[json.loads(l) for l in open("./aw116_output/results.jsonl") if l.strip()]
n=len(recs); s=sum(1 for r in recs if r.get("success"))
print(f"done {n}/116 | success {s} ({s/n*100:.1f}%) | errors {sum(1 for r in recs if r.get('error'))}")
PY
```

---

## 7. 关键文件清单

> 路径相对于本 `eval/` 目录。测试脚本与本文档在 `eval/` 下，与 `android_world/` 同级。

| 文件 | 作用 |
|---|---|
| `run_android_world.py`            | 独立测试脚本（Agent 主循环 + vLLM 客户端 + 动作映射） |
| `README.md`                           | 本文档 |
| `android_world/server/android_server.py`         | AndroidWorld FastAPI server（需改 adb 路径 + 端口 5001，见第 4 节） |
| `~/.local/lib/androidworld-sqlite/libsqlite3.dylib` | 自编译的带 FTS4 的 sqlite，启动 server 时经 `DYLD_LIBRARY_PATH` 加载 |
