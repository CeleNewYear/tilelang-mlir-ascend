#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# TileLang Remote Runner — Configuration & Shared Functions
# ---------------------------------------------------------------------------
# Source this file from remote_verify.sh or remote_run.sh to get the
# connection configuration and helper utilities.
#
# Customize the variables below for your environment.
# ---------------------------------------------------------------------------
set -euo pipefail

# ---- Connection Configuration ----
# REMOTE_HOST: (REQUIRED) target server address, e.g. "root@192.168.1.100".
# Must be set via the TILELANG_REMOTE_HOST environment variable before sourcing this file:
#   export TILELANG_REMOTE_HOST="root@192.168.1.100"
REMOTE_HOST="${TILELANG_REMOTE_HOST:-""}"

# JUMP_HOST: (OPTIONAL) ProxyJump host alias from ~/.ssh/config.
# Leave empty for direct connection.
JUMP_HOST="${TILELANG_JUMP_HOST:-}"

# DOCKER_CONTAINER: (OPTIONAL) running container name on the remote host.
# Leave empty to execute directly on the remote host (no docker).
DOCKER_CONTAINER="${TILELANG_DOCKER_CONTAINER:-}"

# ---- Paths & Defaults ----
REMOTE_BASE_DIR="${TILELANG_REMOTE_BASE_DIR:-/tmp/tl_remote}"
TIMEOUT="${TILELANG_TIMEOUT:-120}"

# Script-relative path to the smoke-test example (run from repo root)
# Default: examples/flash_attn_npuir.py
SMOKE_TEST_SCRIPT="${TILELANG_SMOKE_TEST_SCRIPT:-examples/flash_attn_npuir.py}"

# ---- SSH Options ----
# Security: StrictHostKeyChecking=ask is the safe default to prevent MITM.
# For automated CI/CD, export TILELANG_SSH_OPTS="-o StrictHostKeyChecking=accept-new ..."
# and pre-populate ~/.ssh/known_hosts with the target host keys.
SSH_OPTS="${TILELANG_SSH_OPTS:--o StrictHostKeyChecking=ask -o ConnectTimeout=10 -o ServerAliveInterval=30}"
SCP_OPTS="${TILELANG_SCP_OPTS:--o StrictHostKeyChecking=ask -o ConnectTimeout=10}"

# ---- Runtime Guard: fail fast if REMOTE_HOST is not set ----
if [ -z "${REMOTE_HOST}" ]; then
    echo "ERROR: TILELANG_REMOTE_HOST is not set.  Please export it before running:" >&2
    echo "  export TILELANG_REMOTE_HOST=\"user@host\"" >&2
    exit 1
fi


# ===========================================================================
# Shared Functions
# ===========================================================================

# Build the ssh prefix based on JUMP_HOST.
build_ssh_cmd() {
    if [ -n "${JUMP_HOST}" ]; then
        echo "ssh ${SSH_OPTS} -J ${JUMP_HOST} ${REMOTE_HOST}"
    else
        echo "ssh ${SSH_OPTS} ${REMOTE_HOST}"
    fi
}

# Build the scp prefix based on JUMP_HOST.
build_scp_cmd() {
    if [ -n "${JUMP_HOST}" ]; then
        echo "scp ${SCP_OPTS} -o ProxyJump=${JUMP_HOST}"
    else
        echo "scp ${SCP_OPTS}"
    fi
}

# Build the preload prefix that sources .bashrc inside the container.
# When DOCKER_CONTAINER is set, the container's .bashrc has [ -z "$PS1" ] && return
# which skips CANN/tilelang environment setup in non-interactive shells.
# We export PS1=x to force .bashrc to execute fully.
build_bashrc_preload() {
    if [ -n "${DOCKER_CONTAINER}" ]; then
        echo "export PS1=x; source ~/.bashrc 2>/dev/null;"
    else
        echo ""
    fi
}

# Build the remote execution wrapper.
# If DOCKER_CONTAINER is set, wrap with "docker exec -i <container> bash -c".
# Otherwise, wrap with "bash -c".
build_remote_exec() {
    if [ -n "${DOCKER_CONTAINER}" ]; then
        echo "docker exec -i ${DOCKER_CONTAINER} bash -c"
    else
        echo "bash -c"
    fi
}

# Upload a local file to the remote container (or host if no docker).
# Args: local_path remote_host_path
# When DOCKER_CONTAINER is set, the file lands at remote_host_path inside the
# container via scp→host→docker cp pipeline.
upload_file() {
    local local_path="$1"
    local remote_container_path="$2"
    local ssh_cmd scp_cmd
    ssh_cmd=$(build_ssh_cmd)
    scp_cmd=$(build_scp_cmd)

    if [ -n "${DOCKER_CONTAINER}" ]; then
        # Stage on host first, then docker cp into container.
        local host_tmp="/tmp/tl_upload_$$"
        local escaped_container_path
        escaped_container_path=$(printf %q "${DOCKER_CONTAINER}:${remote_container_path}")
        ${scp_cmd} "${local_path}" "${REMOTE_HOST}:${host_tmp}" >&2
        ${ssh_cmd} "docker cp ${host_tmp} ${escaped_container_path}" >&2
        ${ssh_cmd} "rm -f \"${host_tmp}\"" >&2
    else
        ${scp_cmd} "${local_path}" "${REMOTE_HOST}:${remote_container_path}" >&2
    fi
}

# Create a directory on the remote execution target.
# When DOCKER_CONTAINER is set, creates inside the container.
mk_remote_dir() {
    local path="$1"
    local ssh_cmd remote_exec
    ssh_cmd=$(build_ssh_cmd)
    remote_exec=$(build_remote_exec)
    ${ssh_cmd} "${remote_exec} \"mkdir -p ${path}\"" >&2
}

# Remove a directory on the remote execution target.
rm_remote_dir() {
    local path="$1"
    local ssh_cmd remote_exec
    ssh_cmd=$(build_ssh_cmd)
    remote_exec=$(build_remote_exec)
    ${ssh_cmd} "${remote_exec} \"rm -rf ${path}\"" >&2 2>/dev/null || true
}

# Quick SSH connectivity check.  Returns 0 if OK.
check_ssh_ready() {
    local ssh_cmd
    ssh_cmd=$(build_ssh_cmd)
    if ${ssh_cmd} "echo SSH_OK" 2>/dev/null | grep -q SSH_OK; then
        return 0
    else
        return 1
    fi
}

# Quick docker container check.  Returns 0 if container is running.
# Only call when DOCKER_CONTAINER is non-empty.
check_docker_ready() {
    local ssh_cmd
    ssh_cmd=$(build_ssh_cmd)
    if ${ssh_cmd} "docker inspect -f '{{.State.Running}}' \"${DOCKER_CONTAINER}\"" 2>/dev/null | grep -q true; then
        return 0
    else
        return 1
    fi
}
