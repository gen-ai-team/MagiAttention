#!/usr/bin/env python3
# Copyright (c) 2025-2026 SandAI. All Rights Reserved.
#
# Multi-GPU worker: MagiAttention sequence-parallel vs FA3 head-parallel (Ulysses).
# Launch with torchrun, e.g.:
#   torchrun --standalone --nproc_per_node=2 magi_sp_vs_fa3_ulysses_worker.py --out results/ws2.json
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from einops import rearrange
from flash_attn_interface import flash_attn_func

from magi_attention.api import (
    calc_attn,
    compute_pad_size,
    dispatch,
    magi_attn_flex_key,
    undispatch,
)
from magi_attention.common import AttnRanges
from magi_attention.common.enum import AttnMaskType
from magi_attention.config import DispatchConfig, DistAttnConfig
from magi_attention.meta import make_global_bucket_from_qk_ranges


# ---------------------------------------------------------------------------
# Config defaults (overridable via CLI)
# ---------------------------------------------------------------------------

DEFAULT_SEQLEN = 4096
DEFAULT_NUM_HEADS = 128  # MHA: Hq = Hkv
DEFAULT_HEAD_DIMS = "192,192,128;192,192,192"  # (dq, dk, dv)
DEFAULT_CHUNK_SIZE = 512
DEFAULT_DTYPE = "bfloat16"
DEFAULT_SEED = 42
DEFAULT_WARMUP = 5
DEFAULT_ITERS = 20


def _dtype_from_str(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "fp16": torch.float16}[
        name
    ]


def parse_head_dims(spec: str) -> list[tuple[int, int, int]]:
    """Parse 'dq,dk,dv;dq,dk,dv' into list of triples. Requires dq == dk."""
    out: list[tuple[int, int, int]] = []
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        dq, dk, dv = (int(x.strip()) for x in part.split(","))
        if dq != dk:
            raise ValueError(f"expected dq==dk, got ({dq}, {dk}, {dv})")
        out.append((dq, dk, dv))
    if not out:
        raise ValueError(f"empty --head-dims: {spec!r}")
    return out


def calculate_attn_flops(
    q_ranges: AttnRanges,
    k_ranges: AttnRanges,
    attn_mask_type: list[AttnMaskType],
    total_seqlen_q: int,
    num_heads_q: int,
    head_dim_qk: int,
    head_dim_v: int,
) -> dict[str, float]:
    """Asymmetric-aware FLOPs (same convention as cutedsl notebook)."""
    attn_area = make_global_bucket_from_qk_ranges(
        q_ranges,
        k_ranges,
        attn_mask_type,
        num_chunks=1,
        chunk_size=total_seqlen_q,
    ).area
    # QK^T + PV  →  2 * area * H * (dqk + dv)
    flops_fwd = 2.0 * attn_area * num_heads_q * (head_dim_qk + head_dim_v)
    flops_bwd = flops_fwd * 2.5
    return {"fwd": flops_fwd, "bwd": flops_bwd, "1f1b": flops_fwd + flops_bwd}


def setup_dist() -> tuple[int, int, int, dist.ProcessGroup, torch.device]:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    group = dist.new_group(list(range(world_size)), backend="nccl")
    device = torch.device(f"cuda:{local_rank}")
    return rank, local_rank, world_size, group, device


def sync_barrier() -> None:
    torch.cuda.synchronize()
    dist.barrier()


def max_abs_rel(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    """Max abs error and max relative error on entries with |ref| > 1e-2."""
    diff = (a.float() - b.float()).abs()
    abs_e = float(diff.max())
    ref = b.float().abs()
    significant = ref > 1e-2
    if significant.any():
        rel_e = float((diff[significant] / ref[significant]).max())
    else:
        rel_e = float("nan")
    return abs_e, rel_e


# ---------------------------------------------------------------------------
# FA3 Ulysses (head-parallel) — TE-free
# ---------------------------------------------------------------------------


class _AllToAllSingle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        ctx.group = group
        out = torch.empty_like(x)
        dist.all_to_all_single(out, x.contiguous(), group=group)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        grad_in = torch.empty_like(grad_out)
        dist.all_to_all_single(grad_in, grad_out.contiguous(), group=ctx.group)
        return grad_in, None


def all2all_seq_to_head(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """[T_local, H, D] -> [T_global, H/cp, D]."""
    cp = dist.get_world_size(group)
    if cp <= 1:
        return x
    y = rearrange(x, "t h d -> h t d").contiguous()
    y = _AllToAllSingle.apply(y, group)
    return rearrange(y, "(cp h) t d -> (cp t) h d", cp=cp).contiguous()


def all2all_head_to_seq(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """[T_global, H/cp, D] -> [T_local, H, D]."""
    cp = dist.get_world_size(group)
    if cp <= 1:
        return x
    y = _AllToAllSingle.apply(x.contiguous(), group)
    return rearrange(y, "(cp t) h d -> t (cp h) d", cp=cp).contiguous()


def seq_shard(x: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    t = x.shape[0]
    assert t % world_size == 0, f"seqlen={t} not divisible by world_size={world_size}"
    chunk = t // world_size
    return x[rank * chunk : (rank + 1) * chunk].contiguous()


def seq_unshard(x_local: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    parts = [torch.empty_like(x_local) for _ in range(dist.get_world_size(group))]
    dist.all_gather(parts, x_local.contiguous(), group=group)
    return torch.cat(parts, dim=0)


def fa3_ulysses_fwd_bwd(
    q_local: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    dout_local: torch.Tensor,
    group: dist.ProcessGroup,
    causal: bool,
    softmax_scale: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_local = q_local.detach().requires_grad_(True)
    k_local = k_local.detach().requires_grad_(True)
    v_local = v_local.detach().requires_grad_(True)

    q_h = all2all_seq_to_head(q_local, group)
    k_h = all2all_seq_to_head(k_local, group)
    v_h = all2all_seq_to_head(v_local, group)

    out_h = flash_attn_func(
        q_h.unsqueeze(0),
        k_h.unsqueeze(0),
        v_h.unsqueeze(0),
        softmax_scale=softmax_scale,
        causal=causal,
    )
    assert isinstance(out_h, torch.Tensor)
    out_local = all2all_head_to_seq(out_h.squeeze(0), group)

    out_local.backward(dout_local)
    assert q_local.grad is not None and k_local.grad is not None and v_local.grad is not None
    return out_local.detach(), q_local.grad.detach(), k_local.grad.detach(), v_local.grad.detach()


def fa3_ulysses_out_only(
    q_local: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    group: dist.ProcessGroup,
    causal: bool,
    softmax_scale: float | None,
) -> torch.Tensor:
    q_h = all2all_seq_to_head(q_local, group)
    k_h = all2all_seq_to_head(k_local, group)
    v_h = all2all_seq_to_head(v_local, group)
    out_h = flash_attn_func(
        q_h.unsqueeze(0),
        k_h.unsqueeze(0),
        v_h.unsqueeze(0),
        softmax_scale=softmax_scale,
        causal=causal,
    )
    assert isinstance(out_h, torch.Tensor)
    return all2all_head_to_seq(out_h.squeeze(0), group)


# ---------------------------------------------------------------------------
# MagiAttention SP
# ---------------------------------------------------------------------------


def make_magi_key(
    seqlen: int,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim_qk: int,
    head_dim_v: int,
    world_size: int,
    chunk_size: int,
    group: dist.ProcessGroup,
    mask: str,
):
    q_ranges = AttnRanges.from_ranges([[0, seqlen]])
    k_ranges = AttnRanges.from_ranges([[0, seqlen]])
    attn_mask_type = [
        AttnMaskType.CAUSAL if mask == "causal" else AttnMaskType.FULL
    ]
    pad_size = compute_pad_size(
        total_seqlen_q=seqlen, cp_size=world_size, chunk_size=chunk_size
    )
    dist_attn_config = DistAttnConfig(
        dispatch_config=DispatchConfig(chunk_size=chunk_size),
    )
    key = magi_attn_flex_key(
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_mask_type=attn_mask_type,
        total_seqlen_q=seqlen,
        total_seqlen_k=seqlen,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim_qk,
        head_dim_v=head_dim_v,
        pad_size=pad_size,
        cp_group_or_mesh=group,
        dist_attn_config=dist_attn_config,
    )
    return key, q_ranges, k_ranges, attn_mask_type


def magi_fwd_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    key,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q = q.detach().requires_grad_(True)
    k = k.detach().requires_grad_(True)
    v = v.detach().requires_grad_(True)

    lq = dispatch(q, key=key)
    lk = dispatch(k, key=key)
    lv = dispatch(v, key=key)
    lout, _ = calc_attn(lq, lk, lv, key=key)
    out = undispatch(lout, key=key)
    out.backward(dout)
    assert q.grad is not None and k.grad is not None and v.grad is not None
    return out.detach(), q.grad.detach(), k.grad.detach(), v.grad.detach()


def bench_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    sync_barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync_barrier()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / iters


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_one_case(
    mask: str,
    *,
    seqlen: int,
    num_heads: int,
    head_dim_qk: int,
    head_dim_v: int,
    chunk_size: int,
    dtype: torch.dtype,
    seed: int,
    warmup: int,
    iters: int,
    rank: int,
    world_size: int,
    group: dist.ProcessGroup,
    device: torch.device,
) -> dict[str, Any]:
    num_heads_q = num_heads_kv = num_heads
    assert num_heads_q % world_size == 0
    assert num_heads_kv % world_size == 0
    assert seqlen % world_size == 0

    causal = mask == "causal"
    softmax_scale = 1.0 / math.sqrt(head_dim_qk)

    torch.manual_seed(seed + head_dim_qk * 1009 + head_dim_v * 17)
    q = torch.randn(seqlen, num_heads_q, head_dim_qk, device=device, dtype=dtype)
    k = torch.randn(seqlen, num_heads_kv, head_dim_qk, device=device, dtype=dtype)
    v = torch.randn(seqlen, num_heads_kv, head_dim_v, device=device, dtype=dtype)
    dout = torch.randn(seqlen, num_heads_q, head_dim_v, device=device, dtype=dtype)
    for t in (q, k, v, dout):
        dist.broadcast(t, src=0, group=group)

    key, q_ranges, k_ranges, attn_mask_type = make_magi_key(
        seqlen,
        num_heads_q,
        num_heads_kv,
        head_dim_qk,
        head_dim_v,
        world_size,
        chunk_size,
        group,
        mask,
    )
    flops = calculate_attn_flops(
        q_ranges,
        k_ranges,
        attn_mask_type,
        seqlen,
        num_heads_q,
        head_dim_qk,
        head_dim_v,
    )

    magi_out, magi_dq, magi_dk, magi_dv = magi_fwd_bwd(q, k, v, dout, key)

    q_l = seq_shard(q, rank, world_size)
    k_l = seq_shard(k, rank, world_size)
    v_l = seq_shard(v, rank, world_size)
    dout_l = seq_shard(dout, rank, world_size)
    fa_out_l, fa_dq_l, fa_dk_l, fa_dv_l = fa3_ulysses_fwd_bwd(
        q_l, k_l, v_l, dout_l, group, causal, softmax_scale
    )
    fa_out = seq_unshard(fa_out_l, group)
    fa_dq = seq_unshard(fa_dq_l, group)
    fa_dk = seq_unshard(fa_dk_l, group)
    fa_dv = seq_unshard(fa_dv_l, group)

    corr: dict[str, Any] = {"mask": mask}
    for name, a, b in [
        ("out", magi_out, fa_out),
        ("dq", magi_dq, fa_dq),
        ("dk", magi_dk, fa_dk),
        ("dv", magi_dv, fa_dv),
    ]:
        abs_e, rel_e = max_abs_rel(a, b)
        corr[f"{name}_abs"] = abs_e
        corr[f"{name}_rel"] = rel_e

    lq = dispatch(q.detach(), key=key).detach().requires_grad_(True)
    lk = dispatch(k.detach(), key=key).detach().requires_grad_(True)
    lv = dispatch(v.detach(), key=key).detach().requires_grad_(True)
    ldout = dispatch(dout.detach(), key=key).detach()

    def magi_fwd():
        with torch.no_grad():
            calc_attn(lq.detach(), lk.detach(), lv.detach(), key=key)
        torch.cuda.synchronize()

    def magi_1f1b():
        if lq.grad is not None:
            lq.grad = None
        if lk.grad is not None:
            lk.grad = None
        if lv.grad is not None:
            lv.grad = None
        lout, _ = calc_attn(lq, lk, lv, key=key)
        lout.backward(ldout, retain_graph=True)
        torch.cuda.synchronize()

    q_l = seq_shard(q.detach(), rank, world_size).detach().requires_grad_(True)
    k_l = seq_shard(k.detach(), rank, world_size).detach().requires_grad_(True)
    v_l = seq_shard(v.detach(), rank, world_size).detach().requires_grad_(True)
    dout_l = seq_shard(dout.detach(), rank, world_size).detach()

    def fa_fwd():
        with torch.no_grad():
            fa3_ulysses_out_only(
                q_l.detach(), k_l.detach(), v_l.detach(), group, causal, softmax_scale
            )
        torch.cuda.synchronize()

    def fa_1f1b():
        if q_l.grad is not None:
            q_l.grad = None
        if k_l.grad is not None:
            k_l.grad = None
        if v_l.grad is not None:
            v_l.grad = None
        out_l = fa3_ulysses_out_only(q_l, k_l, v_l, group, causal, softmax_scale)
        out_l.backward(dout_l, retain_graph=True)
        torch.cuda.synchronize()

    sync_barrier()
    magi_fwd_ms = bench_ms(magi_fwd, warmup, iters)
    magi_1f1b_ms = bench_ms(magi_1f1b, warmup, iters)
    fa_fwd_ms = bench_ms(fa_fwd, warmup, iters)
    fa_1f1b_ms = bench_ms(fa_1f1b, warmup, iters)

    def tflops(flops_val: float, ms: float) -> float:
        return flops_val / ms * 1e-9 if ms > 0 else float("nan")

    thr = {
        "magi_fwd_ms": magi_fwd_ms,
        "magi_1f1b_ms": magi_1f1b_ms,
        "fa3_ulysses_fwd_ms": fa_fwd_ms,
        "fa3_ulysses_1f1b_ms": fa_1f1b_ms,
        "magi_fwd_tflops": tflops(flops["fwd"], magi_fwd_ms),
        "magi_1f1b_tflops": tflops(flops["1f1b"], magi_1f1b_ms),
        "fa3_ulysses_fwd_tflops": tflops(flops["fwd"], fa_fwd_ms),
        "fa3_ulysses_1f1b_tflops": tflops(flops["1f1b"], fa_1f1b_ms),
        "work_tflops_fwd": flops["fwd"] * 1e-12,
        "work_tflops_1f1b": flops["1f1b"] * 1e-12,
    }

    return {
        "mask": mask,
        "head_dim_q": head_dim_qk,
        "head_dim_k": head_dim_qk,
        "head_dim_v": head_dim_v,
        "correctness": corr,
        "throughput": thr,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, required=True, help="JSON output path (rank0 writes)")
    p.add_argument("--seqlen", type=int, default=DEFAULT_SEQLEN)
    p.add_argument("--num-heads", type=int, default=DEFAULT_NUM_HEADS)
    p.add_argument(
        "--head-dims",
        type=str,
        default=DEFAULT_HEAD_DIMS,
        help="Semicolon-separated (dq,dk,dv) triples, e.g. '192,192,128;192,192,192'",
    )
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    p.add_argument("--dtype", type=str, default=DEFAULT_DTYPE)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    p.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    p.add_argument(
        "--masks",
        type=str,
        default="full,causal",
        help="Comma-separated masks: full,causal",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size, group, device = setup_dist()
    dtype = _dtype_from_str(args.dtype)
    masks = [m.strip() for m in args.masks.split(",") if m.strip()]
    head_dims = parse_head_dims(args.head_dims)

    if rank == 0:
        print(
            f"[worker] world_size={world_size} seqlen={args.seqlen} "
            f"H={args.num_heads} head_dims={head_dims} "
            f"masks={masks} dtype={args.dtype}"
        )

    results: list[dict[str, Any]] = []
    for dq, dk, dv in head_dims:
        for mask in masks:
            if rank == 0:
                print(f"[worker] running mask={mask} dims=({dq},{dk},{dv}) ...")
            row = run_one_case(
                mask,
                seqlen=args.seqlen,
                num_heads=args.num_heads,
                head_dim_qk=dq,
                head_dim_v=dv,
                chunk_size=args.chunk_size,
                dtype=dtype,
                seed=args.seed,
                warmup=args.warmup,
                iters=args.iters,
                rank=rank,
                world_size=world_size,
                group=group,
                device=device,
            )
            results.append(row)
            if rank == 0:
                c = row["correctness"]
                t = row["throughput"]
                print(
                    f"  corr out_abs={c['out_abs']:.3e} dq_abs={c['dq_abs']:.3e} "
                    f"dk_abs={c['dk_abs']:.3e} dv_abs={c['dv_abs']:.3e}"
                )
                print(
                    f"  thr  magi_fwd={t['magi_fwd_tflops']:.1f}  "
                    f"fa3_fwd={t['fa3_ulysses_fwd_tflops']:.1f}  "
                    f"magi_1f1b={t['magi_1f1b_tflops']:.1f}  "
                    f"fa3_1f1b={t['fa3_ulysses_1f1b_tflops']:.1f} TFLOP/s"
                )

    payload = {
        "world_size": world_size,
        "seqlen": args.seqlen,
        "num_heads": args.num_heads,
        "head_dims": [{"dq": dq, "dk": dk, "dv": dv} for dq, dk, dv in head_dims],
        "chunk_size": args.chunk_size,
        "dtype": args.dtype,
        "seed": args.seed,
        "warmup": args.warmup,
        "iters": args.iters,
        "results": results,
    }

    if rank == 0:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[worker] wrote {out_path.resolve()}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
