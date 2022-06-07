import torch
import torch.nn as nn
from mmdet.models.backbones.swin import WindowMSA, ShiftWindowMSA
from mmdet3d.ops.spconv import SparseConv3d, SubMConv3d
from mmdet3d.models.utils.transformer import MultiheadAttention
from typing import Union
from thop import profile


__all__ = ["flops_counter"]


# TODO: no need to consider ShiftWindowMSA since it contains WindowMSA
def count_window_msa(m: Union[WindowMSA, ShiftWindowMSA], x, y):
    if isinstance(m, WindowMSA):
        embed_dims = m.embed_dims
        num_heads = m.num_heads
    else:
        embed_dims = m.w_msa.embed_dims
        num_heads = m.w_msa.num_heads
    B, N, C = x[0].shape
    # qkv = model.qkv(x)
    m.total_ops += B * N * embed_dims * 3 * embed_dims
    # attn = (q @ k.transpose(-2, -1))
    m.total_ops += B * num_heads * N * (embed_dims // num_heads) * N
    # x = (attn @ v)
    m.total_ops += num_heads * B * N * N * (embed_dims // num_heads)
    # x = m.proj(x)
    m.total_ops += B * N * embed_dims * embed_dims


def count_sparseconv(m: Union[SparseConv3d, SubMConv3d], x, y):
    indice_dict = y.indice_dict[m.indice_key]
    kmap_size = indice_dict[-2].sum().item()
    m.total_ops += kmap_size * x[0].features.shape[1] * y.features.shape[1]


def count_mha(m: Union[MultiheadAttention, nn.MultiheadAttention], x, y):
    flops = 0 
    if len(x) == 3:
        q, k, v = x
    elif len(x) == 2:
        q, k = x
        v = k
    elif len(x) == 1:
        q = x[0]
        k = v = q
    else:
        return

    batch_first = m.batch_first \
        if hasattr(m, 'batch_first') else False
    if batch_first:
        batch_size = q.shape[0]
        len_idx = 1
    else:
        batch_size = q.shape[1]
        len_idx = 0

    dim_idx = 2

    qdim = q.shape[dim_idx]
    kdim = k.shape[dim_idx]
    vdim = v.shape[dim_idx]

    qlen = q.shape[len_idx]
    klen = k.shape[len_idx]
    vlen = v.shape[len_idx]

    num_heads = m.num_heads
    assert qdim == m.embed_dim

    if m.kdim is None:
        assert kdim == qdim
    if m.vdim is None:
        assert vdim == qdim

    flops = 0

    # Q scaling
    flops += qlen * qdim

    # Initial projections
    flops += (
        (qlen * qdim * qdim)  # QW
        + (klen * kdim * kdim)  # KW
        + (vlen * vdim * vdim)  # VW
    )

    if m.in_proj_bias is not None:
        flops += (qlen + klen + vlen) * qdim

    # attention heads: scale, matmul, softmax, matmul
    qk_head_dim = qdim // num_heads
    v_head_dim = vdim // num_heads

    head_flops = (
        (qlen * klen * qk_head_dim)  # QK^T
        + (qlen * klen)  # softmax
        + (qlen * klen * v_head_dim)  # AV
    )

    flops += num_heads * head_flops

    # final projection, bias is always enabled
    flops += qlen * vdim * (vdim + 1)

    flops *= batch_size
    m.total_ops += flops


def flops_counter(model, inputs):
    macs, params = profile(
        model, 
        inputs, 
        custom_ops={
            WindowMSA: count_window_msa,
            #ShiftWindowMSA: count_window_msa,
            SparseConv3d: count_sparseconv,
            SubMConv3d: count_sparseconv,
            MultiheadAttention: count_mha
        },
        verbose=False
    )

    return macs, params
