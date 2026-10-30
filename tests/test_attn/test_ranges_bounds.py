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

from typing import Any

import torch
from cuda.bindings import driver as cu

from magi_attention.api import flex_flash_attn_func

SEQ_LEN = 4096
NUM_RANGES = 10  # < 31, so the first warp of the scheduler already reads past the end


def _ranges_at_page_end(q_ranges: torch.Tensor) -> torch.Tensor:
    """Copy of q_ranges whose last byte is the last mapped byte before an unmapped page."""

    def check(result: tuple) -> Any:
        err, *values = result
        assert err == cu.CUresult.CUDA_SUCCESS, err
        return values[0] if values else None

    device = torch.cuda.current_device()
    prop = cu.CUmemAllocationProp()
    prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device
    granularity = check(
        cu.cuMemGetAllocationGranularity(
            prop, cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
        )
    )
    base = check(cu.cuMemAddressReserve(2 * granularity, 0, 0, 0))
    handle = check(cu.cuMemCreate(granularity, prop, 0))
    check(
        cu.cuMemMap(base, granularity, 0, handle, 0)
    )  # the second page stays unmapped
    access = cu.CUmemAccessDesc()
    access.location = prop.location
    access.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    check(cu.cuMemSetAccess(base, granularity, [access], 1))

    class _Buffer:
        __cuda_array_interface__ = {
            "shape": tuple(q_ranges.shape),
            "typestr": "<i4",
            "data": (
                int(base) + granularity - q_ranges.numel() * q_ranges.element_size(),
                False,
            ),
            "version": 3,
        }

    edge = torch.as_tensor(_Buffer(), device="cuda")
    edge.copy_(q_ranges)
    return edge


def test_q_ranges_at_page_end() -> None:
    """Child process: one FFA forward with q_ranges at the page end; prints OK or FAULT."""
    torch.manual_seed(0)
    bounds = torch.linspace(0, SEQ_LEN, NUM_RANGES + 1).int()
    q_ranges = torch.stack([bounds[:-1], bounds[1:]], dim=1).cuda()
    k_ranges = q_ranges.clone()
    attn_type_map = torch.ones(NUM_RANGES, dtype=torch.int32, device="cuda")
    q, k, v = (
        torch.randn(SEQ_LEN, 4, 128, device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    )
    scale = 128**-0.5
    reference, _ = flex_flash_attn_func(
        q,
        k,
        v,
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_type_map=attn_type_map,
        softmax_scale=scale,
    )
    torch.cuda.synchronize()

    edge_ranges = _ranges_at_page_end(q_ranges)
    out, _ = flex_flash_attn_func(
        q,
        k,
        v,
        q_ranges=edge_ranges,
        k_ranges=k_ranges,
        attn_type_map=attn_type_map,
        softmax_scale=scale,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out, reference, rtol=0, atol=0)
