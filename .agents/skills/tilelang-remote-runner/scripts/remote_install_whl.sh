#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# remote_install_whl.sh — Build whl locally, transfer and install on remote Ascend
# ---------------------------------------------------------------------------
# Usage:
#   bash remote_install_whl.sh
#
# This script:
#   1. Builds the whl package locally using build_wheel.sh
#   2. Uploads the whl to the remote Ascend server
#   3. Installs the whl with pip install --force-reinstall inside the container
#   4. Verifies the installation by importing tilelang and checking the cache module
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../" && pwd)"

# shellcheck source=./remote_config.sh
source "${SCRIPT_DIR}/remote_config.sh"

echo "=========================================="
echo "Phase 1: Building whl locally"
echo "=========================================="

cd "${REPO_ROOT}"

# Check if libtilelang.so exists
if [ ! -f "build/libtilelang.so" ]; then
    echo "Error: build/libtilelang.so not found. Run install_npuir.sh first."
    exit 1
fi

# Build wheel
bash build_wheel.sh

# Find the whl file
WHL_FILE=$(ls -t dist/*.whl 2>/dev/null | head -1)
if [ -z "${WHL_FILE}" ]; then
    echo "Error: No .whl file found in dist/"
    exit 1
fi

WHL_BASENAME=$(basename "${WHL_FILE}")
echo "Built: ${WHL_FILE}"

echo ""
echo "=========================================="
echo "Phase 2: Transferring whl to remote"
echo "=========================================="

# Check SSH
SSH_CMD=$(build_ssh_cmd)
echo "Checking SSH to ${REMOTE_HOST} ..."
if ! check_ssh_ready; then
    echo "ERROR: SSH connection failed."
    exit 1
fi

if [ -n "${DOCKER_CONTAINER}" ]; then
    if ! check_docker_ready; then
        echo "ERROR: Docker container '${DOCKER_CONTAINER}' not running."
        exit 1
    fi
fi

REMOTE_RUN_DIR="${REMOTE_BASE_DIR}/whl_install_$$_$(date +%Y%m%d_%H%M%S)"
REMOTE_EXEC=$(build_remote_exec)
BASHRC_PRELOAD=$(build_bashrc_preload)

echo "Creating remote directory ${REMOTE_RUN_DIR} ..."
mk_remote_dir "${REMOTE_RUN_DIR}"

echo "Uploading ${WHL_BASENAME} ..."
upload_file "${WHL_FILE}" "${REMOTE_RUN_DIR}/${WHL_BASENAME}"

echo ""
echo "=========================================="
echo "Phase 3: Installing whl on remote"
echo "=========================================="

REMOTE_WHL="${REMOTE_RUN_DIR}/${WHL_BASENAME}"
ESCAPED_WHL=$(printf %q "${REMOTE_WHL}")

echo "Pip uninstall existing tilelang (if any) ..."
set +e
UNINSTALL_OUT=$(${SSH_CMD} "${REMOTE_EXEC} \"${BASHRC_PRELOAD} pip uninstall -y tilelang 2>&1\"" 2>&1)
set -e
echo "${UNINSTALL_OUT}"

echo "Pip install ${WHL_BASENAME} with --force-reinstall --no-deps ..."
set +e
INSTALL_OUT=$(${SSH_CMD} "${REMOTE_EXEC} \"${BASHRC_PRELOAD} pip install --force-reinstall --no-deps ${ESCAPED_WHL} 2>&1\"" 2>&1)
INSTALL_EXIT=$?
set -e
echo "${INSTALL_OUT}"

if [ ${INSTALL_EXIT} -ne 0 ]; then
    echo "ERROR: pip install failed with exit code ${INSTALL_EXIT}"
    rm_remote_dir "${REMOTE_RUN_DIR}"
    exit 1
fi

echo ""
echo "=========================================="
echo "Phase 4: Verifying installation"
echo "=========================================="

# Verify that the installed tilelang is ours by checking for a special marker
VERIFY_SCRIPT="
import tilelang
import tilelang.cache
from tilelang.cache.kernel_cache import KernelCache

# Check that the new process-safe features are present
assert hasattr(KernelCache, '_get_staging_root'), 'Missing _get_staging_root'
assert hasattr(KernelCache, '_safe_write_file'), 'Missing _safe_write_file'
assert hasattr(KernelCache, '_is_complete_cache_dir'), 'Missing _is_complete_cache_dir'
assert hasattr(KernelCache, '_get_cache_namespace'), 'Missing _get_cache_namespace'
assert hasattr(KernelCache, '_get_tilelang_lib_stamp'), 'Missing _get_tilelang_lib_stamp'

from tilelang.utils.language import get_prim_func_name
assert callable(get_prim_func_name), 'get_prim_func_name not callable'

# Print a unique marker so we know it's our build
print('CACHE_OPTIMIZATION_V2_ACTIVE')
print(f'tilelang version: {tilelang.__version__}')
print(f'Cache dir: {tilelang.cache.get_cache_dir()}')
print('Installation verified: new process-safe cache is active.')
"

set +e
VERIFY_OUT=$(${SSH_CMD} "${REMOTE_EXEC} \"${BASHRC_PRELOAD} python -c '${VERIFY_SCRIPT}' 2>&1\"" 2>&1)
VERIFY_EXIT=$?
set -e
echo "${VERIFY_OUT}"

# Cleanup
rm_remote_dir "${REMOTE_RUN_DIR}"

if [ ${VERIFY_EXIT} -ne 0 ] || ! echo "${VERIFY_OUT}" | grep -q "CACHE_OPTIMIZATION_V2_ACTIVE"; then
    echo ""
    echo "ERROR: Installation verification FAILED."
    echo "The remote environment may be using a stale tilelang installation."
    echo "Check PYTHONPATH and any existing pip installations."
    exit 1
fi

echo ""
echo "=========================================="
echo "SUCCESS: whl built, transferred, installed, and verified."
echo "=========================================="
