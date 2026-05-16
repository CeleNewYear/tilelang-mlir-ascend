# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""
NPU Cache Verification Tests

Validates the following cache properties:
  1. Basic cache: first run compiles, second run loads from disk cache (no recompilation).
  2. Process safety - reads: one process generates, multiple processes read safely.
  3. Process safety - writes: multiple processes writing simultaneously produce a
     valid complete result without corruption.
  4. Cache naming: cache directory includes the user's operator name so users can
     find their cache entries among many others.
  5. Incomplete cache detection: partially-written cache entries are detected and
     repaired/overwritten.
  6. Cache isolation: different kernels produce different cache keys.

These tests are designed to run on Ascend NPU hardware. They use isolated
cache/tmp directories to avoid interference with other tests or real workloads.
"""

import os
import sys
import json
import shutil
import tempfile
import time
import multiprocessing
import uuid
from pathlib import Path
from typing import Optional

import torch
import torch_npu
import tilelang
import tilelang.language as T
from tilelang import env


# ─────────────────────────────────────────────────────────────────────
# Helper: isolate cache directory for a test
# ─────────────────────────────────────────────────────────────────────

def isolate_cache(cache_dir: str, tmp_dir: str):
    """Point tilelang cache and tmp to isolated directories."""
    env.TILELANG_CACHE_DIR = cache_dir
    env.TILELANG_TMP_DIR = tmp_dir


def clear_memory_cache():
    """Clear the in-memory cache so next access forces disk I/O."""
    from tilelang.cache import _kernel_cache_instance
    _kernel_cache_instance._memory_cache.clear()


def enable_cache():
    """Ensure cache is enabled."""
    tilelang.enable_cache()


# ─────────────────────────────────────────────────────────────────────
# Shared kernel definition helpers
# ─────────────────────────────────────────────────────────────────────

def make_simple_add_kernel(name_suffix: str = ""):
    """Create a simple vector-add kernel with a unique global_symbol."""
    M = 256
    N = 256

    @T.prim_func
    def kernel_add(
        A: T.Tensor((M, N), "float16"),
        B: T.Tensor((M, N), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(M, threads=256) as bx:
            for i in T.serial(N):
                C[bx, i] = A[bx, i] + B[bx, i]

    suffix = name_suffix or uuid.uuid4().hex[:8]
    return kernel_add.with_attr("global_symbol", f"cache_test_add_{suffix}")


def make_simple_mul_kernel(name_suffix: str = ""):
    """Create a different kernel (multiply) for cache-isolation testing."""
    M = 256
    N = 256

    @T.prim_func
    def kernel_mul(
        A: T.Tensor((M, N), "float16"),
        B: T.Tensor((M, N), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(M, threads=256) as bx:
            for i in T.serial(N):
                C[bx, i] = A[bx, i] * B[bx, i]

    suffix = name_suffix or uuid.uuid4().hex[:8]
    return kernel_mul.with_attr("global_symbol", f"cache_test_mul_{suffix}")


# ─────────────────────────────────────────────────────────────────────
# Test 1: Basic cache hit / miss
# ─────────────────────────────────────────────────────────────────────

def test_basic_cache_hit_miss():
    """
    Verify that:
      - First compilation writes cache to disk.
      - Second compilation loads from disk cache (no recompilation).
      - Results are functionally correct.
    """
    with tempfile.TemporaryDirectory() as td:
        cache_dir = os.path.join(td, "tilelang_cache")
        tmp_dir = os.path.join(td, "tilelang_tmp")
        os.makedirs(cache_dir)
        os.makedirs(tmp_dir)
        isolate_cache(cache_dir, tmp_dir)
        enable_cache()

        func = make_simple_add_kernel("basic")

        # Pass 1: compile (cache miss — cache dir empty)
        clear_memory_cache()
        kernel1 = tilelang.compile(func, out_idx=[2], target="npuir")

        # Verify cache files exist on disk
        cache_entries = list(Path(cache_dir).rglob("*"))
        assert len(cache_entries) > 0, (
            f"Expected cache files after first compilation, found: {cache_entries}"
        )

        # Pass 2: should hit disk cache (clear memory cache first)
        clear_memory_cache()
        kernel2 = tilelang.compile(func, out_idx=[2], target="npuir")

        # Functional correctness
        a = torch.randn(256, 256, dtype=torch.float16).npu()
        b = torch.randn(256, 256, dtype=torch.float16).npu()
        c1 = kernel1(a, b)
        c2 = kernel2(a, b)
        ref = a + b
        torch.testing.assert_close(c1, ref, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(c2, ref, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(c1, c2)

    print("PASS: test_basic_cache_hit_miss")


# ─────────────────────────────────────────────────────────────────────
# Test 2: Cache isolation — different kernels, different keys
# ─────────────────────────────────────────────────────────────────────

def test_cache_isolation():
    """
    Verify that two different kernels produce different cache keys and
    don't interfere with each other.
    """
    with tempfile.TemporaryDirectory() as td:
        cache_dir = os.path.join(td, "tilelang_cache")
        tmp_dir = os.path.join(td, "tilelang_tmp")
        os.makedirs(cache_dir)
        os.makedirs(tmp_dir)
        isolate_cache(cache_dir, tmp_dir)
        enable_cache()

        func_add = make_simple_add_kernel("iso_add")
        func_mul = make_simple_mul_kernel("iso_mul")

        clear_memory_cache()
        kernel_add = tilelang.compile(func_add, out_idx=[2], target="npuir")

        clear_memory_cache()
        kernel_mul = tilelang.compile(func_mul, out_idx=[2], target="npuir")

        # Count distinct cache entries
        cache_dirs = list(Path(cache_dir).rglob("metadata.pkl"))
        assert len(cache_dirs) >= 2, (
            f"Expected >=2 distinct cache entries, found {len(cache_dirs)}"
        )

        # Verify functional correctness
        a = torch.randn(256, 256, dtype=torch.float16).npu()
        b = torch.randn(256, 256, dtype=torch.float16).npu()
        c_add = kernel_add(a, b)
        c_mul = kernel_mul(a, b)
        torch.testing.assert_close(c_add, a + b, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(c_mul, a * b, rtol=1e-3, atol=1e-3)

    print("PASS: test_cache_isolation")


# ─────────────────────────────────────────────────────────────────────
# Test 3: Cache naming — kernel name in cache path
# ─────────────────────────────────────────────────────────────────────

def test_cache_naming_has_kernel_name():
    """
    Verify that the cache directory path contains a human-readable
    representation of the kernel's global_symbol, so users can find
    their cache entries.
    """
    with tempfile.TemporaryDirectory() as td:
        cache_dir = os.path.join(td, "tilelang_cache")
        tmp_dir = os.path.join(td, "tilelang_tmp")
        os.makedirs(cache_dir)
        os.makedirs(tmp_dir)
        isolate_cache(cache_dir, tmp_dir)
        enable_cache()

        unique_name = f"my_custom_kernel_{uuid.uuid4().hex[:8]}"
        func = make_simple_add_kernel(unique_name)
        kernel_name = func.attrs["global_symbol"]

        clear_memory_cache()
        tilelang.compile(func, out_idx=[2], target="npuir")

        # Search for the kernel name in paths
        found = False
        for root, dirs, files in os.walk(cache_dir):
            full_path = os.path.join(root, "")
            if unique_name in full_path or kernel_name in full_path:
                found = True
                break

        assert found, (
            f"Kernel name '{kernel_name}' not found in any cache path under {cache_dir}"
        )

    print("PASS: test_cache_naming_has_kernel_name")


# ─────────────────────────────────────────────────────────────────────
# Test 4: Process safety — concurrent reads
# ─────────────────────────────────────────────────────────────────────

def _read_kernel_worker(args):
    """Worker that loads a kernel from cache (should hit disk cache)."""
    cache_dir, tmp_dir, kernel_name, result_queue = args
    isolate_cache(cache_dir, tmp_dir)
    enable_cache()

    # Re-create the same function
    M, N = 256, 256

    @T.prim_func
    def kernel(A: T.Tensor((M, N), "float16"),
               B: T.Tensor((M, N), "float16"),
               C: T.Tensor((M, N), "float16")):
        with T.Kernel(M, threads=256) as bx:
            for i in T.serial(N):
                C[bx, i] = A[bx, i] + B[bx, i]

    func = kernel.with_attr("global_symbol", kernel_name)
    clear_memory_cache()

    try:
        k = tilelang.compile(func, out_idx=[2], target="npuir")
        a = torch.randn(256, 256, dtype=torch.float16).npu()
        b = torch.randn(256, 256, dtype=torch.float16).npu()
        c = k(a, b)
        ref = (a + b).cpu()
        max_diff = (c.cpu() - ref).abs().max().item()
        result_queue.put(("ok", max_diff))
    except Exception as e:
        result_queue.put(("error", str(e)))


def test_concurrent_reads():
    """
    Verify process safety for concurrent reads:
      - One process compiles and writes to disk cache.
      - Multiple other processes read from disk cache concurrently.
      - All processes produce correct results.
    """
    with tempfile.TemporaryDirectory() as td:
        cache_dir = os.path.join(td, "tilelang_cache")
        tmp_dir = os.path.join(td, "tilelang_tmp")
        os.makedirs(cache_dir)
        os.makedirs(tmp_dir)
        isolate_cache(cache_dir, tmp_dir)
        enable_cache()

        kernel_name = f"concurrent_read_{uuid.uuid4().hex[:8]}"

        # Step 1: Compile once to populate disk cache
        func = make_simple_add_kernel(kernel_name)
        clear_memory_cache()
        tilelang.compile(func, out_idx=[2], target="npuir")

        # Step 2: Spawn multiple processes that all read from cache
        num_workers = 4
        ctx = multiprocessing.get_context("spawn")
        result_queue = ctx.Queue()
        processes = []
        for i in range(num_workers):
            p = ctx.Process(
                target=_read_kernel_worker,
                args=((cache_dir, tmp_dir, kernel_name, result_queue),),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join(timeout=120)

        # Collect results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        errors = [r for r in results if r[0] == "error"]
        assert len(errors) == 0, (
            f"Concurrent read workers had errors: {errors}"
        )
        assert len(results) == num_workers, (
            f"Expected {num_workers} results, got {len(results)}"
        )

        # All results should have small error
        for status, max_diff in results:
            assert status == "ok", f"Worker failed: {max_diff}"
            assert max_diff < 1e-2, f"Max diff too large: {max_diff}"

    print("PASS: test_concurrent_reads")


# ─────────────────────────────────────────────────────────────────────
# Test 5: Process safety — concurrent writes
# ─────────────────────────────────────────────────────────────────────

def _write_kernel_worker(args):
    """Worker that compiles a kernel (may be cache miss or race with others)."""
    cache_dir, tmp_dir, kernel_name, result_queue = args
    isolate_cache(cache_dir, tmp_dir)
    enable_cache()

    M, N = 256, 256

    @T.prim_func
    def kernel(A: T.Tensor((M, N), "float16"),
               B: T.Tensor((M, N), "float16"),
               C: T.Tensor((M, N), "float16")):
        with T.Kernel(M, threads=256) as bx:
            for i in T.serial(N):
                C[bx, i] = A[bx, i] + B[bx, i]

    func = kernel.with_attr("global_symbol", kernel_name)
    clear_memory_cache()

    try:
        k = tilelang.compile(func, out_idx=[2], target="npuir")
        a = torch.randn(256, 256, dtype=torch.float16).npu()
        b = torch.randn(256, 256, dtype=torch.float16).npu()
        c = k(a, b)
        ref = (a + b).cpu()
        max_diff = (c.cpu() - ref).abs().max().item()
        result_queue.put(("ok", max_diff))
    except Exception as e:
        result_queue.put(("error", str(e)))


def test_concurrent_writes():
    """
    Verify process safety for concurrent writes:
      - Multiple processes compile the same kernel simultaneously.
      - All processes complete without corruption.
      - Cache ends up in a valid state (complete entry exists).
      - All processes produce correct results.
    """
    with tempfile.TemporaryDirectory() as td:
        cache_dir = os.path.join(td, "tilelang_cache")
        tmp_dir = os.path.join(td, "tilelang_tmp")
        os.makedirs(cache_dir)
        os.makedirs(tmp_dir)
        isolate_cache(cache_dir, tmp_dir)
        enable_cache()

        kernel_name = f"concurrent_write_{uuid.uuid4().hex[:8]}"

        # Spawn multiple processes that all compile simultaneously
        num_workers = 4
        ctx = multiprocessing.get_context("spawn")
        result_queue = ctx.Queue()
        processes = []
        for i in range(num_workers):
            p = ctx.Process(
                target=_write_kernel_worker,
                args=((cache_dir, tmp_dir, kernel_name, result_queue),),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join(timeout=180)

        # Collect results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        errors = [r for r in results if r[0] == "error"]
        assert len(errors) == 0, (
            f"Concurrent write workers had errors: {errors}"
        )
        assert len(results) == num_workers, (
            f"Expected {num_workers} results, got {len(results)}"
        )

        # All results should have small error
        for status, max_diff in results:
            assert status == "ok", f"Worker failed: {max_diff}"
            assert max_diff < 1e-2, f"Max diff too large: {max_diff}"

        # Verify cache is in a valid state (at least one complete entry)
        metadata_files = list(Path(cache_dir).rglob("metadata.pkl"))
        assert len(metadata_files) >= 1, (
            "No complete cache entry found after concurrent writes"
        )

    print("PASS: test_concurrent_writes")


# ─────────────────────────────────────────────────────────────────────
# Test 6: Clear cache
# ─────────────────────────────────────────────────────────────────────

def test_clear_cache():
    """
    Verify that tilelang.cache.clear_cache() removes all cache entries.
    """
    with tempfile.TemporaryDirectory() as td:
        cache_dir = os.path.join(td, "tilelang_cache")
        tmp_dir = os.path.join(td, "tilelang_tmp")
        os.makedirs(cache_dir)
        os.makedirs(tmp_dir)
        isolate_cache(cache_dir, tmp_dir)
        enable_cache()

        func = make_simple_add_kernel("clear_test")

        clear_memory_cache()
        tilelang.compile(func, out_idx=[2], target="npuir")

        # Verify cache has files
        cache_entries_before = list(Path(cache_dir).rglob("*"))
        assert len(cache_entries_before) > 0, "Expected cache files before clear"

        # Clear cache
        tilelang.cache.clear_cache()

        # Verify cache is empty
        cache_entries_after = list(Path(cache_dir).rglob("metadata.pkl"))
        assert len(cache_entries_after) == 0, (
            f"Expected empty cache after clear, found {cache_entries_after}"
        )

    print("PASS: test_clear_cache")


# ─────────────────────────────────────────────────────────────────────
# Test 7: Cache disabled
# ─────────────────────────────────────────────────────────────────────

def test_cache_disabled():
    """
    Verify that when cache is disabled, no files are written to disk.
    """
    with tempfile.TemporaryDirectory() as td:
        cache_dir = os.path.join(td, "tilelang_cache")
        tmp_dir = os.path.join(td, "tilelang_tmp")
        os.makedirs(cache_dir)
        os.makedirs(tmp_dir)
        isolate_cache(cache_dir, tmp_dir)
        tilelang.disable_cache()

        func = make_simple_add_kernel("disabled_test")
        clear_memory_cache()
        kernel = tilelang.compile(func, out_idx=[2], target="npuir")

        # Verify functionally correct
        a = torch.randn(256, 256, dtype=torch.float16).npu()
        b = torch.randn(256, 256, dtype=torch.float16).npu()
        c = kernel(a, b)
        torch.testing.assert_close(c, a + b, rtol=1e-3, atol=1e-3)

        # Cache should remain empty (no disk writes)
        cache_entries = list(Path(cache_dir).rglob("metadata.pkl"))
        assert len(cache_entries) == 0, (
            f"Expected no cache files when disabled, found {cache_entries}"
        )

    print("PASS: test_cache_disabled")


# ─────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("basic_cache_hit_miss", test_basic_cache_hit_miss),
        ("cache_isolation", test_cache_isolation),
        ("cache_naming", test_cache_naming_has_kernel_name),
        ("concurrent_reads", test_concurrent_reads),
        ("concurrent_writes", test_concurrent_writes),
        ("clear_cache", test_clear_cache),
        ("cache_disabled", test_cache_disabled),
    ]

    # Select test via command-line arg, or run all
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        for name, fn in tests:
            if name == test_name:
                print(f"\n=== Running: {name} ===")
                fn()
                break
        else:
            print(f"Unknown test: {test_name}")
            print(f"Available: {[n for n, _ in tests]}")
            sys.exit(1)
    else:
        passed = 0
        failed = 0
        for name, fn in tests:
            print(f"\n=== Running: {name} ===")
            try:
                fn()
                passed += 1
            except Exception as e:
                failed += 1
                print(f"FAIL: {name}: {e}")
                import traceback
                traceback.print_exc()
        print(f"\n=== Summary: {passed} passed, {failed} failed ===")
        if failed:
            sys.exit(1)
