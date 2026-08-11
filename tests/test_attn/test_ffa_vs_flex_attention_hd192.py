# Copyright (c) 2025-2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Correctness tests: MagiAttention FFA vs PyTorch FlexAttention at head_dim=192.

Compares ``flex_flash_attn_func`` against ``torch.nn.attention.flex_attention``
for several FlexAttention ``mask_mod`` patterns:

  * full
  * causal
  * inv_causal
  * bi_causal (diagonal when sq == sk)
  * sliding_window_causal
  * sliding_window_full
  * varlen_full
  * varlen_causal

Uses the eager FlexAttention path (no ``torch.compile``) so each mask's
``BlockMask`` is applied correctly across parameterized cases.

Run::

    pytest tests/test_attn/test_ffa_vs_flex_attention_hd192.py -v
"""

from __future__ import annotations

import math
import unittest
from collections.abc import Callable
from functools import partial
from typing import Any

import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.testing._internal.common_utils import run_tests

from magi_attention.api.functools import infer_attn_mask_from_sliding_window
from magi_attention.common.enum import AttnMaskType
from magi_attention.common.range import AttnRange
from magi_attention.functional import flex_flash_attn_func
from magi_attention.testing import parameterize
from magi_attention.utils import str2seed

MaskMod = Callable[[Any, Any, Any, Any], torch.Tensor]

# FFA SM90 same-dim head_dim support currently tops out at 192.
HEAD_DIM = 192

# Align Flex block size with the packed-document boundaries used below
# (seqlen//4). Avoids a single 128x128 FULL block for S=128.
_FLEX_BLOCK_SIZE = 32

# bf16 tolerances vs FlexAttention (eager / uncompiled math path).
# Eager is used on purpose: torch.compile(flex_attention) can reuse a stale
# BlockMask specialization across parameterized cases and silently diverge.
_FWD_RTOL = 2e-2
_BWD_RTOL = 5e-2
# Absolute floor: diagonal / ultra-sparse masks yield ~0 dq/dk noise.
_ABS_ATOL = 5e-3


def _thd_to_bhsd(x: torch.Tensor) -> torch.Tensor:
    """(S, H, D) -> (1, H, S, D)."""
    return x.transpose(0, 1).unsqueeze(0).contiguous()


def _bhsd_to_thd(x: torch.Tensor) -> torch.Tensor:
    """(1, H, S, D) -> (S, H, D)."""
    return x.squeeze(0).transpose(0, 1).contiguous()


def _ranges_to_tensors(
    q_ranges: list[list[int]] | torch.Tensor,
    k_ranges: list[list[int]] | torch.Tensor,
    attn_type_map: list[int] | torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_t = torch.as_tensor(q_ranges, device=device, dtype=torch.int32)
    k_t = torch.as_tensor(k_ranges, device=device, dtype=torch.int32)
    a_t = torch.as_tensor(attn_type_map, device=device, dtype=torch.int32)
    return q_t, k_t, a_t


def _assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float,
    atol: float,
    test_case: str,
) -> None:
    actual_f = actual.detach().float()
    expected_f = expected.detach().float()
    abs_err = float((actual_f - expected_f).abs().max())
    denom = float(expected_f.abs().max())
    rel_err = abs_err / max(denom, 1e-6)
    if abs_err <= atol or rel_err <= rtol:
        return
    raise AssertionError(
        f"{test_case} max_abs_err={abs_err:.3e} max_rel_err={rel_err:.3e} "
        f"(atol={atol}, rtol={rtol})"
    )


# Top-level mask_mods (not nested closures): FlexAttention / create_block_mask
# can specialize nested functions that share a body with a captured boolean
# branch (e.g. ``if causal``) and then reuse the first specialization.


def _mask_full(b, h, q_idx, kv_idx):
    return q_idx == q_idx


def _mask_causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def _mask_inv_causal(b, h, q_idx, kv_idx):
    return q_idx <= kv_idx


def _mask_bi_causal(b, h, q_idx, kv_idx):
    return q_idx == kv_idx


def _mask_sliding_window_causal(b, h, q_idx, kv_idx, window_left: int):
    return (q_idx >= kv_idx) & (q_idx - kv_idx <= window_left)


def _mask_sliding_window_full(b, h, q_idx, kv_idx, window: int):
    return (q_idx - kv_idx).abs() <= window


def _mask_varlen_full(b, h, q_idx, kv_idx, document_id: torch.Tensor):
    return document_id[q_idx] == document_id[kv_idx]


def _mask_varlen_causal(b, h, q_idx, kv_idx, document_id: torch.Tensor):
    return (document_id[q_idx] == document_id[kv_idx]) & (q_idx >= kv_idx)


def _packed_docs(seqlen: int) -> list[tuple[int, int]]:
    return [
        (0, seqlen // 4),
        (seqlen // 4, 3 * seqlen // 4),
        (3 * seqlen // 4, seqlen),
    ]


def _document_id(seqlen: int, device: torch.device) -> torch.Tensor:
    doc_id = torch.zeros(seqlen, device=device, dtype=torch.int32)
    for i, (start, end) in enumerate(_packed_docs(seqlen)):
        doc_id[start:end] = i
    return doc_id


def _build_mask_case(
    name: str,
    seqlen: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, MaskMod]:
    """Return Magi (q_ranges, k_ranges, attn_type_map) + Flex ``mask_mod``."""
    if name == "full":
        q_ranges = [[0, seqlen]]
        k_ranges = [[0, seqlen]]
        attn_type_map = [AttnMaskType.FULL.to_int_type()]
        return (
            *_ranges_to_tensors(q_ranges, k_ranges, attn_type_map, device),
            _mask_full,
        )

    if name == "causal":
        q_ranges = [[0, seqlen]]
        k_ranges = [[0, seqlen]]
        attn_type_map = [AttnMaskType.CAUSAL.to_int_type()]
        return (
            *_ranges_to_tensors(q_ranges, k_ranges, attn_type_map, device),
            _mask_causal,
        )

    if name == "inv_causal":
        q_ranges = [[0, seqlen]]
        k_ranges = [[0, seqlen]]
        attn_type_map = [AttnMaskType.INVCAUSAL.to_int_type()]
        return (
            *_ranges_to_tensors(q_ranges, k_ranges, attn_type_map, device),
            _mask_inv_causal,
        )

    if name == "bi_causal":
        # For sq == sk, Magi bi_causal collapses to the main diagonal.
        q_ranges = [[0, seqlen]]
        k_ranges = [[0, seqlen]]
        attn_type_map = [AttnMaskType.BICAUSAL.to_int_type()]
        return (
            *_ranges_to_tensors(q_ranges, k_ranges, attn_type_map, device),
            _mask_bi_causal,
        )

    if name == "sliding_window_causal":
        window_left = min(64, seqlen - 1)
        q_ranges_obj, k_ranges_obj, mask_types = infer_attn_mask_from_sliding_window(
            q_range=AttnRange(0, seqlen),
            k_range=AttnRange(0, seqlen),
            window_size=(window_left, 0),
        )
        q_ranges = q_ranges_obj.to_naive_ranges()
        k_ranges = k_ranges_obj.to_naive_ranges()
        attn_type_map = [m.to_int_type() for m in mask_types]
        return (
            *_ranges_to_tensors(q_ranges, k_ranges, attn_type_map, device),
            partial(_mask_sliding_window_causal, window_left=window_left),
        )

    if name == "sliding_window_full":
        window = min(32, seqlen - 1)
        q_ranges_obj, k_ranges_obj, mask_types = infer_attn_mask_from_sliding_window(
            q_range=AttnRange(0, seqlen),
            k_range=AttnRange(0, seqlen),
            window_size=(window, window),
        )
        q_ranges = q_ranges_obj.to_naive_ranges()
        k_ranges = k_ranges_obj.to_naive_ranges()
        attn_type_map = [m.to_int_type() for m in mask_types]
        return (
            *_ranges_to_tensors(q_ranges, k_ranges, attn_type_map, device),
            partial(_mask_sliding_window_full, window=window),
        )

    if name in ("varlen_full", "varlen_causal"):
        # Uneven packed documents; square ranges so Magi bottom-right causal
        # matches FlexAttention's per-token ``q_idx >= kv_idx``.
        docs = _packed_docs(seqlen)
        q_ranges = [[a, b] for a, b in docs]
        k_ranges = [[a, b] for a, b in docs]
        causal = name == "varlen_causal"
        attn_type_map = [
            (
                AttnMaskType.CAUSAL.to_int_type()
                if causal
                else AttnMaskType.FULL.to_int_type()
            )
            for _ in docs
        ]
        document_id = _document_id(seqlen, device)
        mask_mod: MaskMod = partial(
            _mask_varlen_causal if causal else _mask_varlen_full,
            document_id=document_id,
        )
        return (*_ranges_to_tensors(q_ranges, k_ranges, attn_type_map, device), mask_mod)

    raise ValueError(f"Unknown mask case: {name}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestFfaVsFlexAttentionHd192(unittest.TestCase):
    """Compare MagiAttention FFA against FlexAttention masks at head_dim=192."""

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", torch.cuda.current_device())

    @property
    def seed(self) -> int:
        return 42

    def _run_magi(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_ranges: torch.Tensor,
        k_ranges: torch.Tensor,
        attn_type_map: torch.Tensor,
        softmax_scale: float,
    ) -> torch.Tensor:
        out, _ = flex_flash_attn_func(
            q,
            k,
            v,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            softmax_scale=softmax_scale,
        )
        return out

    def _run_flex(
        self,
        q_bhsd: torch.Tensor,
        k_bhsd: torch.Tensor,
        v_bhsd: torch.Tensor,
        mask_mod: MaskMod,
        softmax_scale: float,
    ) -> torch.Tensor:
        """Run FlexAttention; returns output in ``(B, H, S, D)`` layout."""
        seqlen = q_bhsd.shape[-2]
        block_mask = create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=seqlen,
            KV_LEN=seqlen,
            device=q_bhsd.device,
            BLOCK_SIZE=_FLEX_BLOCK_SIZE,
        )
        return flex_attention(
            q_bhsd,
            k_bhsd,
            v_bhsd,
            block_mask=block_mask,
            scale=softmax_scale,
        )

    @parameterize("dtype", [torch.bfloat16, torch.float16])
    @parameterize("seqlen", [128, 256, 512])
    @parameterize("num_heads", [2, 4])
    @parameterize(
        "mask_name",
        [
            "full",
            "causal",
            "inv_causal",
            "bi_causal",
            "sliding_window_causal",
            "sliding_window_full",
            "varlen_full",
            "varlen_causal",
        ],
    )
    def test_fwd_bwd_matches_flex_attention(
        self,
        mask_name: str,
        num_heads: int,
        seqlen: int,
        dtype: torch.dtype,
    ) -> None:
        device = self.device
        torch.manual_seed(self.seed + seqlen + num_heads + str2seed(mask_name) % 997)
        head_dim = HEAD_DIM
        softmax_scale = 1.0 / math.sqrt(head_dim)

        q0 = torch.randn(seqlen, num_heads, head_dim, device=device, dtype=dtype)
        k0 = torch.randn(seqlen, num_heads, head_dim, device=device, dtype=dtype)
        v0 = torch.randn(seqlen, num_heads, head_dim, device=device, dtype=dtype)
        do = torch.randn(seqlen, num_heads, head_dim, device=device, dtype=dtype)

        q_ranges, k_ranges, attn_type_map, mask_mod = _build_mask_case(
            mask_name, seqlen, device
        )

        q_m = q0.clone().detach().requires_grad_(True)
        k_m = k0.clone().detach().requires_grad_(True)
        v_m = v0.clone().detach().requires_grad_(True)
        out_m = self._run_magi(
            q_m, k_m, v_m, q_ranges, k_ranges, attn_type_map, softmax_scale
        )
        out_m.backward(do)
        dq_m, dk_m, dv_m = q_m.grad, k_m.grad, v_m.grad

        q_f = _thd_to_bhsd(q0).detach().requires_grad_(True)
        k_f = _thd_to_bhsd(k0).detach().requires_grad_(True)
        v_f = _thd_to_bhsd(v0).detach().requires_grad_(True)
        out_f_bhsd = self._run_flex(q_f, k_f, v_f, mask_mod, softmax_scale)
        out_f_bhsd.backward(_thd_to_bhsd(do))
        out_f = _bhsd_to_thd(out_f_bhsd.detach())
        dq_f = _bhsd_to_thd(q_f.grad)
        dk_f = _bhsd_to_thd(k_f.grad)
        dv_f = _bhsd_to_thd(v_f.grad)

        test_case = (
            f"[hd={head_dim}][{mask_name}][S={seqlen}][H={num_heads}][{dtype}]"
        )
        for name, actual, expected, rtol in [
            ("out", out_m, out_f, _FWD_RTOL),
            ("dq", dq_m, dq_f, _BWD_RTOL),
            ("dk", dk_m, dk_f, _BWD_RTOL),
            ("dv", dv_m, dv_f, _BWD_RTOL),
        ]:
            _assert_close(
                actual,
                expected,
                rtol=rtol,
                atol=_ABS_ATOL,
                test_case=f"{test_case} => {name}",
            )


if __name__ == "__main__":
    run_tests()
