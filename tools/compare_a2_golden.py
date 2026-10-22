#!/usr/bin/env python3
"""Compare an A2 codegen corpus against the checked-in golden manifest."""
from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path

TMP_RE = re.compile(r"/tmp/[^/\s'\"]+")


def norm_cmd(line: str) -> str:
    try:
        parts = shlex.split(line)
    except ValueError:
        parts = line.split()
    out = []
    for p in parts:
        if p.endswith((".npuir", ".mlir")):
            out.append("<MLIR>")
        elif p.startswith("/tmp/"):
            out.append(TMP_RE.sub("<TMP>", p))
        else:
            out.append(p)
    return " ".join(out)


def norm_mlir(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").splitlines()).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--golden", default="golden/a2")
    ap.add_argument("--mode", default=None)
    args = ap.parse_args()

    golden = Path(args.golden)
    cand = Path(args.candidate)
    manifest = json.loads((golden / "MANIFEST.json").read_text())
    failures: list[tuple[str, str, str, str]] = []
    checks = 0
    for mode, entries in manifest["modes"].items():
        if args.mode and mode != args.mode:
            continue
        for e in entries:
            checks += 1
            g_mlir = golden / e["mlir"]
            c_mlir = cand / mode / e["example"] / Path(e["mlir"]).name
            if not c_mlir.is_file():
                failures.append((mode, e["example"], e["kernel"], "missing MLIR"))
                continue
            if norm_mlir(g_mlir.read_text()) != norm_mlir(c_mlir.read_text()):
                failures.append((mode, e["example"], e["kernel"], "MLIR differs"))
                continue
            if e["command"]:
                g_cmd = norm_cmd((golden / e["command"]).read_text().strip())
                c_cmd_path = cand / mode / e["example"] / Path(e["command"]).name
                if not c_cmd_path.is_file():
                    failures.append((mode, e["example"], e["kernel"], "missing command"))
                    continue
                c_cmd = norm_cmd(c_cmd_path.read_text().strip())
                if g_cmd != c_cmd:
                    failures.append(
                        (mode, e["example"], e["kernel"], f"command differs\n  golden: {g_cmd}\n  cand:   {c_cmd}")
                    )

    print(f"checked {checks} kernels, failures {len(failures)}")
    for mode, example, kernel, why in failures[:50]:
        print(f"[{mode}] {example}::{kernel}: {why}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
