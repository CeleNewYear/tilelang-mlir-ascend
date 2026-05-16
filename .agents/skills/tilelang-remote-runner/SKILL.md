---
name: tilelang-remote-runner
description: TileLang npuir 远端验证基础设施。用户提及远端运行、验证 kernel、remote run、remote verify、push to ascend、在昇腾上跑、远端执行、ascend verify 时必须使用本技能。负责将 Agent 生成的 kernel 代码通过 SSH 上传到远程 Ascend 服务器并执行，回传运行结果供 LLM 分析和迭代修复。
---

# TileLang Remote Runner

## Mandatory routing rule

Before answering, follow AGENTS.md section "Docs Auto Routing Rules (Mandatory)".

## What this skill provides

- 远端环境就绪检测（冒烟测试）
- Agent 生成的 kernel 代码上传到远端 Ascend 服务器执行
- stdout / stderr / exit code 完整回传
- 支持注入调试环境变量（如 `TILELANG_DUMP_IR`）
- 自动适配：跳板机 / 直连、docker 容器 / 裸机
- 迭代验证循环：运行 → 错误分析 → 修复 → 重试

## 工作流

### Phase 1 — 环境就绪检测

用户必须保证远端 Python 环境已就绪（tilelang 已安装，torch_npu 可用，NPU 设备可访问）。
脚本不做任何环境初始化（不 source CANN、不 export 任何变量）。

```bash
# 验证远端环境是否可用
bash .agents/skills/tilelang-remote-runner/scripts/remote_verify.sh
```

成功输出：`ENV_READY=true`
失败输出：`ENV_READY=false` + 原始错误信息。

### Phase 2 — Kernel 验证循环

1. Agent 生成 kernel 代码 → 写入 `testing/npuir/remote_verified/` 目录
2. 调用 `remote_run.sh` 上传并执行
3. 分析回传的 stdout / stderr / exit code
4. 若有错误，使用 **tilelang-error-fixer** 诊断修复
5. 用 `replace_in_file` 更新 kernel 代码
6. 重复步骤 2-5 直到 exit code 为 0

Agent 写入规则：
- 文件名格式：`gen_<op_name>_v<iteration>.py`（如 `gen_matmul_v1.py`）
- 每轮修复后递增版本号或原地修改
- 目录 `testing/npuir/remote_verified/` 在用户 IDE 中可见

### Phase 3 — 调试增强（按需）

当需要深入调试时，可通过 `-e` 参数注入环境变量：

```bash
# Dump IR 用于分析 pass 行为
bash .agents/skills/tilelang-remote-runner/scripts/remote_run.sh \
    -e TILELANG_DUMP_IR=1 \
    testing/npuir/remote_verified/gen_kernel.py

# 指定 Expert / Developer 模式
bash .agents/skills/tilelang-remote-runner/scripts/remote_run.sh \
    -e TILELANG_ASCEND_MODE=Expert \
    testing/npuir/remote_verified/gen_kernel.py

# 组合
bash .agents/skills/tilelang-remote-runner/scripts/remote_run.sh \
    -e TILELANG_DUMP_IR=1 \
    -e TILELANG_ASCEND_MODE=Developer \
    testing/npuir/remote_verified/gen_kernel.py
```

## 脚本说明

| 脚本 | 用途 |
|------|------|
| `scripts/remote_config.sh` | 远端连接配置 + 共享函数。用户可修改此文件适配自己的环境 |
| `scripts/remote_install_whl.sh` | 本地构建 whl → 上传 → 远端 pip install → 验证安装 |
| `scripts/remote_verify.sh` | 环境就绪检测：上传 `examples/flash_attn_npuir.py` 并执行 |
| `scripts/remote_run.sh` | 通用远端执行：`remote_run.sh [-e KEY=VAL]... <kernel_path> [extra_args]` |

### 文件传输链路

使用 Docker 容器时，文件通过 **scp → host → docker cp** 三步管道送入容器：

```
本地文件 ──scp──→ 远端宿主机 /tmp/tl_upload_$$ ──docker cp──→ 容器内目标路径
```

不使用 Docker 时，直接 `scp` 到远端宿主机路径。

### remote_config.sh 可配置项

```bash
# 可通过环境变量覆盖。REMOTE_HOST 必须先 export 再执行脚本：
#   export TILELANG_REMOTE_HOST="root@192.168.1.100"
REMOTE_HOST="${TILELANG_REMOTE_HOST:-""}"     # 必需：目标服务器地址（通过环境变量设置）；默认为空，须由用户设置
JUMP_HOST="${TILELANG_JUMP_HOST:-""}"   # 可选：跳板机 Host 别名（示例值 A3_proxy），为空直连
DOCKER_CONTAINER="${TILELANG_DOCKER_CONTAINER:-""}"  # 可选：容器名（示例值如 docker_name），为空在宿主机执行
REMOTE_BASE_DIR="${TILELANG_REMOTE_BASE_DIR:-/tmp/tl_remote}"  # 远端临时工作目录
TIMEOUT="${TILELANG_TIMEOUT:-120}"            # 执行超时（秒）
SMOKE_TEST_SCRIPT="${TILELANG_SMOKE_TEST_SCRIPT:-examples/flash_attn_npuir.py}"  # 冒烟测试脚本
```

### 环境加载机制

Docker 容器内的 `.bashrc` 包含 `[ -z "$PS1" ] && return`，导致 non-interactive shell 跳过 CANN/tilelang 环境设置。脚本通过 `build_bashrc_preload()` 在每次远程执行前注入 `export PS1=x; source ~/.bashrc 2>/dev/null;`，确保环境与手动 `docker exec -it` 完全一致，无需额外配置 `REMOTE_PYTHONPATH`。

### 四种场景自动适配

| JUMP_HOST | DOCKER_CONTAINER | SSH 链路 |
|-----------|------------------|----------|
| 有值 | 有值 | `ssh -J` → `docker exec -i` |
| 空 | 有值 | `ssh` → `docker exec -i` |
| 有值 | 空 | `ssh -J` → `bash -c` |
| 空 | 空 | `ssh` → `bash -c` |

### remote_run.sh 输出格式

```
===REMOTE_EXIT_CODE===<n>
===REMOTE_STDOUT===
<stdout content>
===REMOTE_STDERR===
<stderr content>
```

## 错误分类与路由

| 错误类型 | 路由 Skill | 典型操作 |
|----------|-----------|----------|
| SSH 连通失败 | 直接报告用户 | 检查网络 / 跳板机 / 目标服务器 |
| Docker 容器未运行 | 直接报告用户 | 检查容器状态 |
| 编译失败 | **tilelang-error-fixer** + **tilelang-mlir-skill** | 检查 API 用法、pass 配置 |
| 运行时 crash / core dump | **tilelang-error-fixer** + **tilelang-debug-helper** | 加 IR dump 定位 |
| 精度不达标 | **tilelang-error-fixer** | 检查 dtype、算法逻辑 |
| 超时（疑似死锁） | **tilelang-error-fixer** | 检查 sync_block_set/wait 配对 |
| 冒烟测试自身失败 | 报告用户 | 远端环境可能不完整 |

## Official docs to consult

- docs/快速入门.md
- docs/安装指南.md
- docs/developer/EnvironmentVariables.md
- docs/Tilelang算子调试指南.md

## Related skills

- tilelang-error-fixer
- tilelang-debug-helper
- tilelang-mlir-skill
- tilelang-npuir-overview
- tilelang-vector-skill
- tilelang-cube-skill
- tilelang-mixcv-skill
