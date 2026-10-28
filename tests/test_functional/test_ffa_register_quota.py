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


import itertools
import unittest

from magi_attention.functional._flex_flash_attn_jit import _ffa_register_quota


def _kernel_budget(num_mma_wgs: int) -> int:
    # FlashAttnBwdSm90: NumLoadWarpGroups * ProducerRegs + NumConsumerWarpGroups * ConsumerRegs
    # <= kAvgRegsPerThread * kTotalWarpGroups, kAvgRegsPerThread = 65536 / (WGs * 128) rounded down to 8
    num_wgs = 1 + num_mma_wgs
    return (65536 // (num_wgs * 128) // 8 * 8) * num_wgs


class TestFFARegisterQuota(unittest.TestCase):
    def test_bwd_quota_fits_the_kernel_budget(self):
        for (
            (head_dim, head_dim_v),
            inner_loop_k,
            (block_sparse, index_sparse),
            tma,
        ) in itertools.product(
            [(64, 64), (128, 128), (192, 192), (192, 128)],
            [False, True],
            [(False, False), (True, False), (False, True)],
            [False, True],
        ):
            with self.subTest(
                hd=head_dim,
                hdv=head_dim_v,
                inner_loop_k=inner_loop_k,
                sparse=(block_sparse, index_sparse),
                tma=tma,
            ):
                producer, consumer = _ffa_register_quota(
                    direction="bwd",
                    head_dim=head_dim,
                    kblock_m=None,
                    swap_ab=False,
                    bwd_inner_loop_k=inner_loop_k,
                    block_sparse=block_sparse,
                    index_sparse=index_sparse,
                    sparse_dx_tma_reduce=tma,
                    head_dim_v=head_dim_v,
                )
                num_mma_wgs = (
                    2
                    if inner_loop_k
                    else (3 if head_dim == 192 and head_dim_v == 192 else 2)
                )
                self.assertLessEqual(
                    producer + num_mma_wgs * consumer, _kernel_budget(num_mma_wgs)
                )

    def test_bwd_dense_hd192_quota_matches_fa3(self):
        # 40 + 3 x 152 spilled the MMA accumulators; FA3 uses 32 + 3 x 160 for its 3-WG hd192 backward
        quota = _ffa_register_quota(
            direction="bwd",
            head_dim=192,
            kblock_m=None,
            swap_ab=False,
            bwd_inner_loop_k=False,
            block_sparse=False,
            index_sparse=False,
            sparse_dx_tma_reduce=False,
        )
        self.assertEqual(quota, (32, 160))


if __name__ == "__main__":
    unittest.main()
