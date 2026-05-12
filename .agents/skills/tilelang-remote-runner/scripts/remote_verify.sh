#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# remote_verify.sh — Ascend NPU Remote Environment Smoke Test
# ---------------------------------------------------------------------------
# Usage:
#   bash .agents/skills/tilelang-remote-runner/scripts/remote_verify.sh
#
# This script uploads examples/flash_attn_npuir.py to the remote server and
# runs it.  Success means the remote environment (NPU driver, torch_npu,
# tilelang, CANN) is correctly set up.
#
# Output:
#   ENV_READY=true          — smoke test passed
#   ENV_READY=false         — smoke test failed (error details in stderr)
# ---------------------------------------------------------------------------
set -euo pipefail

# Resolve script directory to locate remote_config.sh regardless of CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../" && pwd)"

# shellcheck source=./remote_config.sh
source "${SCRIPT_DIR}/remote_config.sh"

echo "[remote_verify] Checking SSH connectivity to ${REMOTE_HOST} ..." >&2
if ! check_ssh_ready; then
    echo "ENV_READY=false" >&2
    echo "ERROR: SSH connection to ${REMOTE_HOST} failed." >&2
    exit 1
fi

if [ -n "${DOCKER_CONTAINER}" ]; then
    echo "[remote_verify] Checking docker container '${DOCKER_CONTAINER}' ..." >&2
    if ! check_docker_ready; then
        echo "ENV_READY=false" >&2
        echo "ERROR: Docker container '${DOCKER_CONTAINER}' is not running on ${REMOTE_HOST}." >&2
        exit 1
    fi
fi

# Verify local smoke-test script exists.
SMOKE_LOCAL="${REPO_ROOT}/${SMOKE_TEST_SCRIPT}"
if [ ! -f "${SMOKE_LOCAL}" ]; then
    echo "ENV_READY=false" >&2
    echo "ERROR: Smoke-test script not found: ${SMOKE_LOCAL}" >&2
    exit 1
fi

# Prepare remote work directory.
REMOTE_RUN_DIR="${REMOTE_BASE_DIR}/verify_$$_$(date +%Y%m%d_%H%M%S)"
SSH_CMD=$(build_ssh_cmd)
REMOTE_EXEC=$(build_remote_exec)

echo "[remote_verify] Creating remote directory ${REMOTE_RUN_DIR} ..." >&2
mk_remote_dir "${REMOTE_RUN_DIR}"

SMOKE_BASENAME=$(basename "${SMOKE_TEST_SCRIPT}")
SMOKE_REMOTE="${REMOTE_RUN_DIR}/${SMOKE_BASENAME}"

echo "[remote_verify] Uploading ${SMOKE_TEST_SCRIPT} ..." >&2
upload_file "${SMOKE_LOCAL}" "${SMOKE_REMOTE}"

echo "[remote_verify] Running smoke test on remote ..." >&2

# Preload .bashrc to get CANN/tilelang environment (same as docker exec -it).
BASHRC_PRELOAD=$(build_bashrc_preload)

ESCAPED_SMOKE=$(printf %q "${SMOKE_BASENAME}")
 set +e
 REMOTE_OUTPUT=$(
    ${SSH_CMD} "${REMOTE_EXEC} \"timeout ${TIMEOUT} bash -c '${BASHRC_PRELOAD} cd ${REMOTE_RUN_DIR} && python ${ESCAPED_SMOKE}' 2>&1\"" 2>&1
 )
 EXIT_CODE=$?
 set -e

# Cleanup remote directory.
rm_remote_dir "${REMOTE_RUN_DIR}"

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "ENV_READY=true"
else
    echo "ENV_READY=false"
    echo "===REMOTE_STDOUT==="
    echo "${REMOTE_OUTPUT}"
    echo "===REMOTE_STDERR==="
    echo "===REMOTE_EXIT_CODE===${EXIT_CODE}"
    echo "ERROR: Smoke test failed. Check remote environment (CANN, torch_npu, tilelang)." >&2
    exit 1
fi
