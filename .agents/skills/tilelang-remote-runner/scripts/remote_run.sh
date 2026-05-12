#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# remote_run.sh — Execute a kernel on Ascend NPU remote server
# ---------------------------------------------------------------------------
# Usage:
#   bash remote_run.sh [-e KEY=VAL]... <kernel_path> [extra_args...]
#
# Examples:
#   bash remote_run.sh testing/npuir/remote_verified/gen_matmul.py
#   bash remote_run.sh -e TILELANG_DUMP_IR=1 gen_kernel.py
#   bash remote_run.sh -e TILELANG_DUMP_IR=1 -e TILELANG_ASCEND_MODE=Expert gen.py
#
# Output format:
#   ===REMOTE_EXIT_CODE===<n>
#   ===REMOTE_STDOUT===
#   <stdout content>
#   ===REMOTE_STDERR===
#   <stderr content>
# ---------------------------------------------------------------------------
set -euo pipefail

# Resolve script directory to locate remote_config.sh regardless of CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../" && pwd)"

# shellcheck source=./remote_config.sh
source "${SCRIPT_DIR}/remote_config.sh"

# ---- Parse -e KEY=VAL options ----
ENV_PREFIX=""
ENV_ARRAY=()
KERNEL_PATH=""
EXTRA_ARGS=()

# Validate -e argument: must be KEY=VALUE where KEY is [A-Za-z_][A-Za-z0-9_]*
# Returns 0 on success, 1 on failure.
validate_env_arg() {
    local arg="$1"
    local key="${arg%%=*}"
    local value="${arg#*=}"
    # Reject if no '=' present or key is empty or key is the whole string (no value after =)
    if [ "$arg" = "$key" ] || [ -z "$key" ]; then
        return 1
    fi
    # KEY must match [A-Za-z_][A-Za-z0-9_]*
    if ! [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
        return 1
    fi
    # Value must not be empty after trimming (optional: allow empty value)
    # We allow empty value: KEY=
    return 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -e)
            if [ $# -lt 2 ]; then
                echo "ERROR: -e requires KEY=VAL argument" >&2
                exit 1
            fi
            if ! validate_env_arg "$2"; then
                echo "ERROR: Invalid -e argument: '$2'. Must be KEY=VAL where KEY matches [A-Za-z_][A-Za-z0-9_]*" >&2
                exit 1
            fi
            local_key="${2%%=*}"
            local_val="${2#*=}"
            safe_val="$(printf '%q' "$local_val")"
            ENV_ARRAY+=("${local_key}=${safe_val}")
            ENV_PREFIX="${ENV_PREFIX} ${local_key}=${safe_val}"
            shift 2
            ;;
        -*)
            echo "ERROR: Unknown option: $1" >&2
            echo "Usage: remote_run.sh [-e KEY=VAL]... <kernel_path> [extra_args...]" >&2
            exit 1
            ;;
        *)
            if [ -z "${KERNEL_PATH}" ]; then
                KERNEL_PATH="$1"
            else
                EXTRA_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

if [ -z "${KERNEL_PATH}" ]; then
    echo "ERROR: No kernel file path provided." >&2
    echo "Usage: remote_run.sh [-e KEY=VAL]... <kernel_path> [extra_args...]" >&2
    exit 1
fi

# Resolve kernel path: allow both absolute and relative (to REPO_ROOT).
if [[ "${KERNEL_PATH}" == /* ]]; then
    KERNEL_LOCAL="${KERNEL_PATH}"
else
    KERNEL_LOCAL="${REPO_ROOT}/${KERNEL_PATH}"
fi

if [ ! -f "${KERNEL_LOCAL}" ]; then
    echo "ERROR: Kernel file not found: ${KERNEL_LOCAL}" >&2
    exit 1
fi

# ---- Quick connectivity check ----
echo "[remote_run] Checking SSH to ${REMOTE_HOST} ..." >&2
if ! check_ssh_ready; then
    echo "===REMOTE_EXIT_CODE===1"
    echo "===REMOTE_STDOUT==="
    echo "===REMOTE_STDERR==="
    echo "SSH connection failed: $(build_ssh_cmd)" >&2
    exit 1
fi

if [ -n "${DOCKER_CONTAINER}" ]; then
    if ! check_docker_ready; then
        echo "===REMOTE_EXIT_CODE===1"
        echo "===REMOTE_STDOUT==="
        echo "===REMOTE_STDERR==="
        echo "Docker container '${DOCKER_CONTAINER}' is not running on ${REMOTE_HOST}" >&2
        exit 1
    fi
fi

# ---- Prepare remote execution ----
REMOTE_RUN_DIR="${REMOTE_BASE_DIR}/run_$$_$(date +%Y%m%d_%H%M%S)"
SSH_CMD=$(build_ssh_cmd)
REMOTE_EXEC=$(build_remote_exec)

KERNEL_BASENAME=$(basename "${KERNEL_LOCAL}")

echo "[remote_run] Creating remote directory ${REMOTE_RUN_DIR} ..." >&2
mk_remote_dir "${REMOTE_RUN_DIR}"

echo "[remote_run] Uploading ${KERNEL_BASENAME} ..." >&2
upload_file "${KERNEL_LOCAL}" "${REMOTE_RUN_DIR}/${KERNEL_BASENAME}"

# Build the remote command.
ESCAPED_EXTRA=""
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    for arg in "${EXTRA_ARGS[@]}"; do
        ESCAPED_EXTRA="${ESCAPED_EXTRA} $(printf '%q' "$arg")"
    done
    ESCAPED_EXTRA="${ESCAPED_EXTRA# }"
fi

echo "[remote_run] Executing: ${ENV_PREFIX} python ${KERNEL_BASENAME} ${ESCAPED_EXTRA}" >&2
echo "[remote_run] (timeout=${TIMEOUT}s) ..." >&2

# Preload .bashrc to get CANN/tilelang environment (same as docker exec -it).
BASHRC_PRELOAD=$(build_bashrc_preload)
ESCAPED_KERNEL=$(printf %q "${KERNEL_BASENAME}")

# ---- Execute with timeout ----
# Use bash -c inside timeout because cd/&&/redirects need a shell.
set +e
REMOTE_OUTPUT=$(
    ${SSH_CMD} "${REMOTE_EXEC} \"timeout ${TIMEOUT} bash -c '${BASHRC_PRELOAD} cd ${REMOTE_RUN_DIR} && ${ENV_PREFIX} python ${ESCAPED_KERNEL} ${ESCAPED_EXTRA}' 2>&1\"" 2>&1
)
EXIT_CODE=$?
set -e

# ---- Separate stdout and stderr ----
# The remote exec pipes 2>&1 so all output is in REMOTE_OUTPUT.
# We report it as stdout; stderr is captured only for local SSH errors.
# Remote python stderr is already merged into REMOTE_OUTPUT.
STDERR_CONTENT=""

# Detect timeout (exit code 124 from timeout command).
if [ ${EXIT_CODE} -eq 124 ]; then
    STDERR_CONTENT="TIMEOUT: Kernel execution exceeded ${TIMEOUT} seconds. Possible deadlock or infinite loop."
fi

# ---- Cleanup remote directory ----
rm_remote_dir "${REMOTE_RUN_DIR}"

# ---- Format output ----
echo "===REMOTE_EXIT_CODE===${EXIT_CODE}"
echo "===REMOTE_STDOUT==="
echo "${REMOTE_OUTPUT}"
echo "===REMOTE_STDERR==="
if [ -n "${STDERR_CONTENT}" ]; then
    echo "${STDERR_CONTENT}"
fi