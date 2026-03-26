"""
Function to generate bank of masks for OmniAttention and some auxiliary functions.

pip install "drawsvg~=2.0" to install drawsvg library.
"""

from random import randint

import drawsvg as dw
import numpy as np
import torch
from torch import Tensor


def generate_random_sequence(
    target_len: int = 32000,
    text_cstr: tuple[int, int] = (10, 1024),
    size_cstr: tuple[int, int] = (128, 1024),
    max_frames: int = 61,
    max_retries: int = 10,
) -> tuple[list[list[int]], list[list[int]]]:
    """Generate a random sequence of sub-sequences and their types."""
    full_seq_len = 0
    seq = []
    seq_types = []
    num_retries = max_retries
    while full_seq_len < target_len:
        sub_seq, sub_seq_types = [], []
        mode = randint(0, 4)  # TTI, TTV, TITI, TITV
        t1 = randint(*text_cstr)
        t2 = randint(min(2 * t1, text_cstr[1]), text_cstr[1])
        h = randint(*size_cstr) // 16
        w = randint(*size_cstr) // 16
        f = randint(1, max_frames)
        if mode == 0:  # TT
            sub_seq = [t1 + 2 + t2 + 1]
            sub_seq_types = [1]  # 1 - causal, 2 - full
        if mode == 1:  # TTI
            sub_seq = [t1 + 3 + t2 + 2, h * w, 1]
            sub_seq_types = [1, 2, 1]  # 1 - causal, 2 - full
        if mode == 2:  # TTV
            sub_seq = [t1 + 3 + t2 + 2, h * w * f, 1]
            sub_seq_types = [1, 2, 1]  # 1 - causal, 2 - full
        if mode == 3:  # TITI
            sub_seq = [t1 + 3, h * w, 2 + t2 + 2, h * w, 1]
            sub_seq_types = [1, 2, 1, 2, 1]  # 1 - causal, 2 - full
        if mode == 4:  # TITV
            sub_seq = [t1 + 3, h * w, 2 + t2 + 2, h * w * f, 1]
            sub_seq_types = [1, 2, 1, 2, 1]  # 1 - causal, 2 - full
        sub_seq_len = sum(sub_seq)

        if full_seq_len + sub_seq_len < target_len:
            seq.append(sub_seq)
            seq_types.append(sub_seq_types)
            full_seq_len += sub_seq_len
        else:
            num_retries -= 1

        if num_retries == 0:
            break
    return seq, seq_types


def draw_full_mask(seq: list[list[int]], seq_types: list[list[int]], bsize: int = 64) -> dw.Drawing:
    """Draw a full mask from a sequence of sub-sequences and their types."""
    sub_seq_lens = [sum(s) for s in seq]
    full_len = sum(sub_seq_lens)
    full_len = ((full_len // bsize) + 1) * bsize if full_len % bsize > 0 else full_len

    img = dw.Drawing(full_len, full_len, origin="top-left")
    element = dw.Rectangle(0, 0, full_len, full_len, fill="blue")
    img.append(element)

    sp = (0, 0)  # start point
    for i, ss in enumerate(seq):
        sst = seq_types[i]
        points = [sp[0], sp[1]]
        sp1 = sp
        for j, s in enumerate(ss):
            if sst[j] > 1:
                points += [sp1[0] + s, sp1[1], sp1[0] + s, sp1[1] + s]
            else:
                points += [sp1[0] + s, sp1[1] + s]
            sp1 = (sp1[0] + s, sp1[1] + s)
        points += [sp[0], sp[1] + sub_seq_lens[i]]
        sp = sp1
        element = dw.Lines(*points, stroke="yellow", fill="yellow", close="true")
        img.append(element)
    return img


def reorder_sequence(seq: list[list[int]], seq_types: list[list[int]]) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Reorder a sequence of sub-sequences and their types."""
    text_seq = []
    vis_seq = []
    num_texts = sum(sum(1 for t in st if t == 1) for st in seq_types)
    n_sub_seq = sum(len(s) for s in seq)
    mat1 = np.zeros((n_sub_seq, n_sub_seq), dtype=np.uint8)  # матрица смежности, 0 - пусто, 1 - каузально, 2 - полно
    mat2 = np.zeros((n_sub_seq, n_sub_seq), dtype=np.uint8)  # матрица смежности после перестановки
    sp = 0
    tp = 0  # start point for texts
    vp = num_texts  # start point for vis
    for i, ss in enumerate(seq):
        sst = seq_types[i]
        l_ss = len(ss)
        mat1[sp : sp + l_ss, sp : sp + l_ss] = np.tril(2 * np.ones((l_ss, l_ss), dtype=np.uint8))
        jt, jv = 0, 0  # row counters for text and vis
        for j in range(l_ss):
            kt, kv = 0, 0  # column counters for text and vis
            for k in range(j + 1):
                if j == k:
                    if sst[j] > 1:
                        mat2[vp + jv, vp + kv] = 2
                    else:
                        mat2[tp + jt, tp + kt] = 1
                elif sst[j] == 1 and sst[k] == 1:
                    mat2[tp + jt, tp + kt] = 2
                elif sst[j] == 1 and sst[k] == 2:
                    mat2[tp + jt, vp + kv] = 2
                elif sst[j] == 2 and sst[k] == 1:
                    mat2[vp + jv, tp + kt] = 2
                else:
                    mat2[vp + jv, vp + kv] = 2
                if sst[k] > 1:
                    kv += 1
                else:
                    kt += 1
            if sst[j] > 1:
                vis_seq.append(ss[j])
                jv += 1
            else:
                text_seq.append(ss[j])
                mat1[sp + j, sp + j] = 1
                jt += 1
        tp += jt
        vp += jv
        sp += l_ss
    reordered_seq = text_seq + vis_seq
    return reordered_seq, mat1, mat2


def draw_reordered_full_mask(r_seq: list[int], mat: np.ndarray, bsize: int = 64) -> dw.Drawing:
    """Draw a reordered full mask from a sequence of sub-sequences and their types."""
    num_sub_seq = len(r_seq)
    full_len = sum(r_seq)
    full_len = ((full_len // bsize) + 1) * bsize if full_len % bsize > 0 else full_len
    offsets = np.cumsum([0] + r_seq)

    img = dw.Drawing(full_len, full_len, origin="top-left")
    back = dw.Rectangle(0, 0, full_len, full_len, fill="blue")
    img.append(back)

    for i in range(num_sub_seq):
        for j in range(num_sub_seq):
            if mat[i, j] == 1:
                points = [
                    offsets[j],
                    offsets[i],
                    offsets[j] + r_seq[j],
                    offsets[i] + r_seq[i],
                    offsets[j],
                    offsets[i] + r_seq[i],
                ]
                line = dw.Lines(*points, stroke="green", fill="yellow", close="true")
                img.append(line)
            if mat[i, j] == 2:
                square = dw.Rectangle(offsets[j], offsets[i], r_seq[j], r_seq[i], stroke="green", fill="yellow")
                img.append(square)
    return img


def eq(b1: Tensor, b2: Tensor) -> bool:
    """Check if two blocks are equal."""
    return (b1.abs() - b2.abs()).sum() == 0


def append_block(block: Tensor, unique_blocks: list[Tensor], unique_sums: list[int], block_num: int) -> tuple[int, int]:
    """Append a block to the unique_blocks and return the block number and index."""
    s = block.sum()
    unique = True
    for i, b in enumerate(unique_blocks):
        if s != unique_sums[i]:
            continue
        if eq(block, b):
            unique = False
            ind = i
            break
    if unique:
        unique_blocks.append(block)
        unique_sums.append(s)
        ind = block_num
        block_num += 1
    return block_num, ind


def add_block_to_dict(keys: list[tuple[int, int]], block: Tensor, block_dict: dict[tuple[int, int], Tensor]) -> None:
    """Add a block to the block_dict."""
    for k in keys:
        if k in block_dict:
            block_dict[k] += block
        else:
            block_dict[k] = block.clone()


def process_rectangle(
    top: int,
    left: int,
    bottom: int,
    right: int,
    block_mask: Tensor,
    block_dict: dict[tuple[int, int], Tensor],
    bsize: int = 64,
) -> None:
    """Process a rectangle and add its blocks to the block_dict."""
    t_, l_ = top // bsize, left // bsize
    b_ = bottom // bsize if bottom % bsize > 0 else bottom // bsize - 1
    r_ = right // bsize if right % bsize > 0 else right // bsize - 1

    # Inner part
    block_mask[(t_ + 1) : b_, (l_ + 1) : r_] = 1

    ti = (torch.arange(bsize * t_, bsize * t_ + bsize, dtype=torch.int32) >= top).unsqueeze(1)
    li = (torch.arange(bsize * l_, bsize * l_ + bsize, dtype=torch.int32) >= left).unsqueeze(0)
    bi = (torch.arange(bsize * b_, bsize * b_ + bsize, dtype=torch.int32) < bottom).unsqueeze(1)
    ri = (torch.arange(bsize * r_, bsize * r_ + bsize, dtype=torch.int32) < right).unsqueeze(0)

    # Horizontal edges
    if t_ == b_:
        # Top and bottom edges coincide
        block = (ti * bi).tile(1, bsize).to(torch.uint8)
        keys = [(t_, v) for v in range(l_ + 1, r_)]
        add_block_to_dict(keys, block, block_dict)
    else:
        block = ti.tile(1, bsize).to(torch.uint8)
        keys = [(t_, v) for v in range(l_ + 1, r_)]
        add_block_to_dict(keys, block, block_dict)

        block = bi.tile(1, bsize).to(torch.uint8)
        keys = [(b_, v) for v in range(l_ + 1, r_)]
        add_block_to_dict(keys, block, block_dict)

    # Vertical edges
    if l_ == r_:
        # Left and right edges coincide
        block = (li * ri).tile(bsize, 1).to(torch.uint8)
        keys = [(v, l_) for v in range(t_ + 1, b_)]
        add_block_to_dict(keys, block, block_dict)
    else:
        block = li.tile(bsize, 1).to(torch.uint8)
        keys = [(v, l_) for v in range(t_ + 1, b_)]
        add_block_to_dict(keys, block, block_dict)

        block = ri.tile(bsize, 1).to(torch.uint8)
        keys = [(v, r_) for v in range(t_ + 1, b_)]
        add_block_to_dict(keys, block, block_dict)

    tl_block = (li * ti).to(torch.uint8)
    tr_block = (ri * ti).to(torch.uint8)
    bl_block = (li * bi).to(torch.uint8)
    br_block = (ri * bi).to(torch.uint8)

    if t_ == b_ and l_ == r_:
        # All 4 vertices coincide
        add_block_to_dict([(b_, r_)], tl_block * tr_block * bl_block * br_block, block_dict)
    elif t_ == b_ and l_ != r_:
        # Two horizontal vertices
        add_block_to_dict([(b_, l_)], tl_block * bl_block, block_dict)
        add_block_to_dict([(b_, r_)], tr_block * br_block, block_dict)
    elif t_ != b_ and l_ == r_:
        # Two vertical vertices
        add_block_to_dict([(t_, l_)], tl_block * tr_block, block_dict)
        add_block_to_dict([(b_, r_)], bl_block * br_block, block_dict)
    else:
        # All vertices are different
        add_block_to_dict([(t_, l_)], tl_block, block_dict)
        add_block_to_dict([(t_, r_)], tr_block, block_dict)
        add_block_to_dict([(b_, l_)], bl_block, block_dict)
        add_block_to_dict([(b_, r_)], br_block, block_dict)


def process_triangle(
    top: int,
    left: int,
    bottom: int,
    right: int,
    block_mask: Tensor,
    block_dict: dict[tuple[int, int], Tensor],
    bsize: int = 64,
) -> None:
    """Process a triangle and add its blocks to the block_dict."""
    t_, l_ = top // bsize, left // bsize
    b_ = bottom // bsize if bottom % bsize > 0 else bottom // bsize - 1
    r_ = right // bsize if right % bsize > 0 else right // bsize - 1

    # Inner part and diagonal
    if b_ - t_ > 1 and r_ - l_ > 1:
        block_mask[(t_ + 1) : b_, (l_ + 1) : r_] = torch.tril(
            torch.ones((b_ - t_ - 1, r_ - l_ - 1), dtype=torch.uint8)
        ) + torch.eye(b_ - t_ - 1, dtype=torch.uint8)

    ti = (torch.arange(bsize * t_, bsize * t_ + bsize, dtype=torch.int32) >= top).unsqueeze(1)
    li = (torch.arange(bsize * l_, bsize * l_ + bsize, dtype=torch.int32) >= left).unsqueeze(0)
    bi = (torch.arange(bsize * b_, bsize * b_ + bsize, dtype=torch.int32) < bottom).unsqueeze(1)
    ri = (torch.arange(bsize * r_, bsize * r_ + bsize, dtype=torch.int32) < right).unsqueeze(0)

    # Horizontal edge
    block = bi.tile(1, bsize).to(torch.uint8)
    keys = [(b_, v) for v in range(l_ + 1, r_)]
    add_block_to_dict(keys, block, block_dict)

    # Vertical edge
    block = li.tile(bsize, 1).to(torch.uint8)
    keys = [(v, l_) for v in range(t_ + 1, b_)]
    add_block_to_dict(keys, block, block_dict)

    bl_block = (li * bi).to(torch.uint8)
    tl_block = torch.tril(li * ti).to(torch.uint8)
    br_block = torch.tril(ri * bi).to(torch.uint8)

    if t_ == b_ and l_ == r_:
        # All 3 vertices coincide
        add_block_to_dict([(b_, r_)], tl_block * bl_block * br_block, block_dict)
    else:
        # All vertices are different
        add_block_to_dict([(t_, l_)], tl_block, block_dict)
        add_block_to_dict([(b_, l_)], bl_block, block_dict)
        add_block_to_dict([(b_, r_)], br_block, block_dict)


def create_mask_bank(r_seq: list[int], mat: np.ndarray, bsize: int = 64) -> tuple[Tensor, list[Tensor]]:
    """Create a mask bank from a reordered sequence of sub-sequences and adjacency matrix."""
    num_sub_seq = len(r_seq)
    full_len = sum(r_seq)
    num_blocks = full_len // bsize + 1 if full_len % bsize > 0 else full_len // bsize
    full_len = num_blocks * bsize
    of = torch.cumsum(torch.IntTensor([0] + r_seq), 0).numpy().tolist()  # segment offsets

    unique_blocks = [
        torch.zeros((bsize, bsize), dtype=torch.uint8),
        torch.ones((bsize, bsize), dtype=torch.uint8),
        torch.tril(torch.ones((bsize, bsize), dtype=torch.uint8)),
    ]
    unique_sums = [0, bsize**2, unique_blocks[-1].to(torch.uint16).sum()]
    block_dict: dict[tuple[int, int], Tensor] = {}
    block_num = 3
    block_mask = torch.zeros((num_blocks, num_blocks), dtype=torch.int32)
    for i in range(num_sub_seq):
        for j in range(num_sub_seq):
            if mat[i, j] == 2:
                process_rectangle(of[i], of[j], of[i + 1], of[j + 1], block_mask, block_dict)
            if mat[i, j] == 1:
                process_triangle(of[i], of[j], of[i + 1], of[j + 1], block_mask, block_dict)

    for key, block in block_dict.items():
        block_num, ind = append_block(block, unique_blocks, unique_sums, block_num)
        block_mask[key[0], key[1]] = ind

    return block_mask, unique_blocks


def qk_segments(reordered_seq: list[int], mat: np.ndarray) -> tuple[Tensor, Tensor, Tensor]:
    """QK segments descripion for SDPA, Flex or MagiAttention masks."""
    num_sub_seq = len(reordered_seq)
    offsets = np.cumsum([0] + reordered_seq)

    q_segments = []
    k_segments = []
    seg_types = []

    start_seg = 0
    for i in range(num_sub_seq - 1):
        if (mat[i, i] == 1 and mat[i + 1, i] == 0) or (mat[i + 1, i + 1] == 2):
            q_segments.append([start_seg, offsets[i + 1]])
            k_segments.append([start_seg, offsets[i + 1]])
            seg_types.append(1)
            start_seg = offsets[i + 1]
        if mat[i + 1, i + 1] == 2:
            break
    rect_start = i + 1

    for i in range(num_sub_seq):
        start_seg = -1
        js = rect_start if i < rect_start else 0
        for j in range(js, num_sub_seq):
            if mat[i, j] == 2 and start_seg == -1:
                start_seg = offsets[j]
            if mat[i, j] == 0 and start_seg >= 0:
                q_segments.append([offsets[i], offsets[i + 1]])
                k_segments.append([start_seg, offsets[j]])
                seg_types.append(0)
                start_seg = -1
        if start_seg > 0:
            q_segments.append([offsets[i], offsets[i + 1]])
            k_segments.append([start_seg, offsets[num_sub_seq]])
            seg_types.append(0)

    q_segments = torch.Tensor(q_segments).to(torch.int32)
    k_segments = torch.Tensor(k_segments).to(torch.int32)
    seg_types = torch.Tensor(seg_types).to(torch.int32)
    return q_segments, k_segments, seg_types


def make_sdpa_mask(q_segments: Tensor, k_segments: Tensor, seg_types: Tensor, bsize: int = 64) -> Tensor:
    """Make a SDPA mask from QK segments."""
    size = torch.maximum(q_segments.max(), k_segments.max())
    size = ((size // bsize) + 1) * bsize if size % bsize > 0 else size
    mask = torch.zeros((size, size), dtype=torch.bool)
    for i in range(q_segments.shape[0]):
        if seg_types[i] == 0:
            mask[q_segments[i, 0] : q_segments[i, 1], k_segments[i, 0] : k_segments[i, 1]] = True
        else:
            mask[q_segments[i, 0] : q_segments[i, 1], k_segments[i, 0] : k_segments[i, 1]] = torch.tril(
                torch.ones((q_segments[i, 1] - q_segments[i, 0], k_segments[i, 1] - k_segments[i, 0]), dtype=torch.bool)
            )
    return mask
