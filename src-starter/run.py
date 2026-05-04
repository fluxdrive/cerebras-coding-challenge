#!/usr/bin/env cs_python
"""Host runner for Top-K k-NN kernel."""

import argparse
import os, sys
import numpy as np
from cerebras.sdk.runtime.sdkruntimepybind import (
    SdkRuntime, MemcpyDataType, MemcpyOrder
)

# Add challenge root to path so we can import reference oracle
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from reference import topk_reference, ALL_CASES

# ── case name → maker function ─────────────────────────────────────────────
CASE_MAP = {
    "baseline":   ALL_CASES[0],
    "k_eq_1":     ALL_CASES[1],
    "k_large":    ALL_CASES[2],
    "uneven":     ALL_CASES[3],
    "all_equal":  ALL_CASES[4],
    "duplicates": ALL_CASES[5],
}

# ── args ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True, help="compiled output dir")
parser.add_argument("--case", required=True, choices=list(CASE_MAP))
parser.add_argument("--cmaddr", default=None, help="IP:port for real hardware")
args = parser.parse_args()

# ── build test case ────────────────────────────────────────────────────────
case = CASE_MAP[args.case]()
D    = case["D"].astype(np.float32)   # (N, d)
q    = case["q"].astype(np.float32)   # (d,)
K    = int(case["K"])
P    = int(case["P"])

N, d = D.shape
P2   = P * P
rows_per_pe = int(np.ceil(N / P2))

# ── compute oracle answer ──────────────────────────────────────────────────
ref_idx, ref_dist = topk_reference(D, q, K)

# ── pad D to exactly P2 * rows_per_pe rows (last PE may be short) ─────────
N_padded = P2 * rows_per_pe
if N_padded > N:
    pad = np.full((N_padded - N, d), np.inf, dtype=np.float32)
    D_padded = np.vstack([D, pad])
else:
    D_padded = D

# ── shard D across PEs ────────────────────────────────────────────────────
# PE(px, py) has linear index py*P + px and owns rows [linear*rows_per_pe : +rows_per_pe]
# memcpy ROW_MAJOR layout: [py, px, rows_per_pe * d_dim]
# We need to reshape into (P_height, P_width, rows_per_pe * d) for memcpy_h2d
D_shards = D_padded.reshape(P2, rows_per_pe, d)  # (P2, rows_per_pe, d)
# Reorder to (P_height=P, P_width=P, rows_per_pe*d) row-major
D_host = D_shards.reshape(P, P, rows_per_pe * d)  # (Ph, Pw, l)
D_flat = D_host.flatten().astype(np.float32)

# q is broadcast to all PEs — replicate for memcpy
q_host = np.tile(q, P2).reshape(P, P, d)
q_flat = q_host.flatten().astype(np.float32)

# ── run ────────────────────────────────────────────────────────────────────
runner = SdkRuntime(args.name, cmaddr=args.cmaddr)

D_sym    = runner.get_id("D_shard")
q_sym    = runner.get_id("q")
idx_sym  = runner.get_id("out_idx")
dist_sym = runner.get_id("out_dist")

runner.load()
runner.run()

# Copy D shards to each PE
runner.memcpy_h2d(D_sym, D_flat, 0, 0, P, P, rows_per_pe * d,
                  streaming=False,
                  data_type=MemcpyDataType.MEMCPY_32BIT,
                  order=MemcpyOrder.ROW_MAJOR,
                  nonblock=False)

# Copy q to each PE
runner.memcpy_h2d(q_sym, q_flat, 0, 0, P, P, d,
                  streaming=False,
                  data_type=MemcpyDataType.MEMCPY_32BIT,
                  order=MemcpyOrder.ROW_MAJOR,
                  nonblock=False)

# Launch kernel
runner.launch("run", nonblock=False)

# Read back results from PE(0,0) only
out_idx_raw  = np.zeros(K, dtype=np.int32)
out_dist_raw = np.zeros(K, dtype=np.float32)

runner.memcpy_d2h(out_idx_raw, idx_sym, 0, 0, 1, 1, K,
                  streaming=False,
                  data_type=MemcpyDataType.MEMCPY_32BIT,
                  order=MemcpyOrder.ROW_MAJOR,
                  nonblock=False)

runner.memcpy_d2h(out_dist_raw, dist_sym, 0, 0, 1, 1, K,
                  streaming=False,
                  data_type=MemcpyDataType.MEMCPY_32BIT,
                  order=MemcpyOrder.ROW_MAJOR,
                  nonblock=False)

runner.stop()


# ── verify ─────────────────────────────────────────────────────────────────
idx_match  = np.array_equal(out_idx_raw, ref_idx)
dist_match = np.allclose(out_dist_raw, ref_dist, atol=1e-3, rtol=1e-3)

if not idx_match:
    print(f"FAIL: {args.case} — index mismatch")
    print(f"  got:      {out_idx_raw[:8].tolist()}")
    print(f"  expected: {ref_idx[:8].tolist()}")

if not dist_match:
    print(f"FAIL: {args.case} — distance mismatch")
    print(f"  got:      {out_dist_raw[:8].tolist()}")
    print(f"  expected: {ref_dist[:8].tolist()}")

if idx_match and dist_match:
    print(f"PASS: {args.case}")



