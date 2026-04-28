# Adapted from https://github.com/tile-ai/tilelang/blob/e666d2d3cc483829c57618c9ebf2e4f4ada0819d/examples/deepseek_v32/sparse_mla_fwd.py
import math
import torch
import tilelang
from tilelang import language as T


@tilelang.jit(out_idx=[-2, -1])
def sparse_mla_fwd(
    heads,
    dim,
    tail_dim,
    topk,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    CP0=True,
    block_I=64,
    num_stages=2,
    threads=256,
):
    assert dim == tilelang.math.next_power_of_2(dim), (
        f"haven't check padding correctness yet, dim={dim}"
    )
    assert tail_dim == tilelang.math.next_power_of_2(tail_dim), (
        f"haven't check padding correctness yet, dim={tail_dim}"
    )
    assert is_causal == True, "non-casual is not supported"
    assert topk % block_I == 0, (
        "otherwise will load some index=0 thus causing wrong kv to be loaded"
    )
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5 * 1.44269504  # log2(e)
    else:
        sm_scale = sm_scale * 1.44269504  # log2(e)

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    head_kv = heads // kv_group
    q_shape = [batch, seq_len, heads, dim + tail_dim]
    kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]
    o_shape = [batch, seq_len, heads, dim]
    indices_shape = [batch, seq_len, kv_group, topk]
    lse_shape = [batch, seq_len, heads]
    indices_dtype = T.int32
    dtype = T.bfloat16
    accum_dtype = T.float32

    G = kv_group
    H = head_kv
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    if padded_H != H:
        assert kv_group == 1, (
            "here we solve the H padding automatically, other wise you should handle Q copy and Output copy with your mask (when kv_group == 1, use g_i * padded_H:(g_i+1) * padded_H would be handled automatically)"
        )
    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    D = dim
    D_tail = tail_dim

    if head_kv > 64:
        assert head_kv % 64 == 0, "head_kv should be a multiple of 64"
        REPLICATE_H = head_kv // 64
    else:
        REPLICATE_H = 1

    H_per_block = padded_H if REPLICATE_H == 1 else 64

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        KV: T.Tensor(kv_shape, dtype),  # type: ignore
        Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
        Output: T.Tensor(o_shape, dtype),  # type: ignore
        Lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len * REPLICATE_H, is_npu=True) as (cid, _):
            bx = cid
            for by in T.serial(batch):
                for bz in T.serial(kv_group):
                    Q_shared = T.alloc_shared([H_per_block, D], dtype)
                    Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype)
                    KV_shared = T.alloc_shared([BI, D], dtype)
                    K_tail_shared = T.alloc_shared([BI, D_tail], dtype)
                    O_shared = T.alloc_shared([H_per_block, D], dtype)
                    Lse_shared = T.alloc_shared([H_per_block], accum_dtype)
                    mask = T.alloc_fragment([BI], "bool")

                    acc_o = T.alloc_fragment([H_per_block, D], accum_dtype)
                    acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
                    S_shared = T.alloc_shared([H_per_block, BI], dtype)
                    sumexp = T.alloc_fragment([H_per_block], accum_dtype)
                    sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
                    alpha = T.alloc_fragment([H_per_block], accum_dtype)
                    m_i = T.alloc_fragment([H_per_block], accum_dtype)
                    m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)

                    T.fill(acc_o, 0)
                    T.fill(sumexp, 0)
                    T.fill(m_i, -(2**30))  # avoid -inf - inf to cause nan

                    b_i, g_i = by, bz
                    s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)
                    q_i = s_i
                    max_kv_i = q_i

                    H0 = g_i * padded_H + (
                        0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64
                    )
                    H1 = H0 + H_per_block

                    T.copy(Q[b_i, s_i, H0:H1, :D], Q_shared)
                    T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_shared)

                    for i_i in T.Pipelined(NI, num_stages=num_stages):
                        for bi_i in T.Parallel(BI):
                            # Changed here for thd
                            mask[bi_i] = Indices[b_i, s_i, g_i, i_i * BI + bi_i] != -1

                        for bi_i, d_i in T.Parallel(BI, D):
                            KV_shared[bi_i, d_i] = KV[
                                b_i, Indices[b_i, s_i, g_i, i_i * BI + bi_i], g_i, d_i
                            ]
                        for bi_i, d_i in T.Parallel(BI, D_tail):
                            K_tail_shared[bi_i, d_i] = KV[
                                b_i,
                                Indices[b_i, s_i, g_i, i_i * BI + bi_i],
                                g_i,
                                D + d_i,
                            ]

                        for h_i, bi_i in T.Parallel(H_per_block, BI):
                            acc_s[h_i, bi_i] = T.if_then_else(
                                mask[bi_i], 0, -T.infinity(acc_s.dtype)
                            )
                        T.gemm(
                            Q_shared,
                            KV_shared,
                            acc_s,
                            transpose_B=True,
                            size=[H_per_block, D, BI],
                        )
                        T.gemm(
                            Q_tail_shared,
                            K_tail_shared,
                            acc_s,
                            transpose_B=True,
                            size=[H_per_block, D_tail, BI],
                        )
                        T.copy(m_i, m_i_prev)
                        T.reduce_max(acc_s, m_i, dim=1, clear=False)
                        for h_i in T.Parallel(H_per_block):
                            m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
                        for h_i in T.Parallel(H_per_block):
                            alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                        for h_i, bi_i in T.Parallel(H_per_block, BI):
                            acc_s[h_i, bi_i] = T.exp2(
                                acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale
                            )
                        T.reduce_sum(
                            acc_s, sumexp_i, dim=1
                        )  # is this a accumulate operator?
                        for h_i in T.Parallel(H_per_block):
                            sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                        for h_i, d_i in T.Parallel(H_per_block, D):
                            acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha[h_i]

                        T.copy(acc_s, S_shared)
                        T.gemm(S_shared, KV_shared, acc_o)

                    # Rescale
                    for h_i, d_i in T.Parallel(H_per_block, D):
                        acc_o[h_i, d_i] /= sumexp[h_i]
                    for h_i in T.Parallel(H_per_block):
                        sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale

                    T.copy(acc_o, Output[b_i, s_i, H0:H1, :])
                    T.copy(sumexp, Lse[b_i, s_i, H0:H1])

    return main


def sparse_mla_fwd_interface(
    q,
    kv,
    indices,
    sm_scale=None,
    return_p_sum: bool = False,
    d_v=512,
    block_I=64,
    num_stages=2,
    threads=256,
):
    q = q.unsqueeze(0)
    kv = kv.unsqueeze(0)
    indices = indices.unsqueeze(0)

    is_casual = True
    assert return_p_sum == False, "This kernel file is for fwd only"
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    batch, seq_len, heads, dim_plus_tail_dim = q.shape
    _, seq_len_kv, kv_group, _ = kv.shape

    assert dim_plus_tail_dim == 576, "you should assign dim otherwise"
    dim = d_v

    assert kv.shape[-1] == dim_plus_tail_dim
    tail_dim = dim_plus_tail_dim - dim
    assert kv.shape[0] == batch
    _, _, _, topk = indices.shape
    assert indices.shape == (batch, seq_len, kv_group, topk)

    kernel = sparse_mla_fwd(
        heads,
        dim,
        tail_dim,
        topk,
        kv_group,
        sm_scale,
        is_casual,
        block_I=block_I,
        num_stages=num_stages,
        threads=threads,
    )
    out, lse = kernel(q, kv, indices)
    out = out.squeeze(0)
    lse = lse.squeeze(0)
    return out, lse


def sparse_mla_fwd_reference(
    q,
    kv,
    indices,
    dim,
    tail_dim,
    kv_group=1,
    sm_scale=None,
):
    """PyTorch reference implementation for sparse_mla_fwd.

    Args:
        q: [seq_len, heads, dim + tail_dim]
        kv: [seq_len_kv, kv_group, dim + tail_dim]
        indices: [seq_len, kv_group, topk], -1 means masked
        dim: dimension of V (and the head part of Q/K)
        tail_dim: dimension of the tail part of Q/K
        kv_group: number of KV groups
        sm_scale: softmax scale (natural log base), None for default

    Returns:
        out: [seq_len, heads, dim]
        lse: [seq_len, heads] -- log-sum-exp in log2 base
    """
    seq_len, heads, total_dim = q.shape
    seq_len_kv, G, _ = kv.shape
    _, _, topk = indices.shape

    assert total_dim == dim + tail_dim
    assert heads % kv_group == 0
    assert kv_group == G

    head_kv = heads // kv_group

    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5 * 1.44269504  # log2(e)
    else:
        sm_scale = sm_scale * 1.44269504  # log2(e)

    # Convert to natural log scale for PyTorch softmax
    scale_e = sm_scale * math.log(2.0)

    acc_dtype = torch.float32
    q_dtype = q.dtype

    Qf = q.to(acc_dtype)
    KVf = kv.to(acc_dtype)
    indices_long = indices.long()

    out = torch.zeros(seq_len, heads, dim, device=q.device, dtype=q_dtype)
    lse = torch.zeros(seq_len, heads, device=q.device, dtype=acc_dtype)

    for s in range(seq_len):
        for g in range(kv_group):
            h0 = g * head_kv
            h1 = (g + 1) * head_kv

            # Q for this position and group: [head_kv, dim+tail_dim]
            q_s = Qf[s, h0:h1, :]  # [head_kv, total_dim]

            # Gather KV at sparse indices
            idx = indices_long[s, g, :]  # [topk]

            # Determine valid indices (not -1 and satisfy causal: <= s)
            valid = (idx >= 0) & (idx < seq_len_kv)

            # Prepare full valid index list
            valid_idx = idx[valid]

            if valid_idx.numel() == 0:
                out[s, h0:h1, :] = 0
                lse[s, h0:h1] = float("-inf")
                continue

            # Gather KV: [num_valid, dim+tail_dim]
            k = KVf[valid_idx, g, :]  # [num_valid, total_dim]
            v = KVf[valid_idx, g, :dim]  # [num_valid, dim]

            # Attention logits: [head_kv, num_valid]
            logits = q_s @ k.transpose(0, 1)  # [head_kv, num_valid]

            # Causal mask: kv position must be <= query position s
            causal_mask = valid_idx <= s  # [num_valid]
            logits = logits.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))

            # Check if any valid after causal
            any_valid = causal_mask.any()
            if not any_valid:
                out[s, h0:h1, :] = 0
                lse[s, h0:h1] = float("-inf")
                continue

            # Scale to natural log base and softmax
            logits_e = logits * scale_e  # [head_kv, num_valid]
            probs = torch.softmax(logits_e, dim=-1)  # [head_kv, num_valid]

            # Weighted sum of V
            out_s = probs.to(acc_dtype) @ v.to(acc_dtype)  # [head_kv, dim]
            out[s, h0:h1, :] = out_s.to(q_dtype)

            # Log-sum-exp in log2 base (to match kernel's Lse output)
            lse[s, h0:h1] = torch.logsumexp(logits_e, dim=-1) / math.log(2.0)

    return out, lse


def run_test():
    """Test sparse_mla_fwd kernel against PyTorch reference."""
    device = "npu:0"

    B = 1  # batch, interface adds this internally
    Sq = 32  # seq_len
    Skv = 32  # seq_len_kv
    H = 16  # heads
    D = 128  # dim (V dimension)
    D_tail = 64  # tail_dim
    topk = 16  # top-k sparse indices
    kv_group = 1
    block_I = 16

    q = torch.randn(Sq, H, D + D_tail, dtype=torch.float16, device=device)
    kv = torch.randn(Skv, kv_group, D + D_tail, dtype=torch.float16, device=device)

    # Generate random sparse indices (causal: each query attends to positions <= itself)
    indices_cpu = torch.zeros(Sq, kv_group, topk, dtype=torch.int32)
    for s in range(Sq):
        for g in range(kv_group):
            max_pos = min(s, Skv - 1)
            if max_pos >= 0:
                idx = torch.randint(0, max_pos + 1, (topk,), dtype=torch.int32)
                indices_cpu[s, g, :] = idx
            else:
                indices_cpu[s, g, :] = -1
    indices = indices_cpu.to(device)

    print(f"Q shape: {q.shape}")
    print(f"KV shape: {kv.shape}")
    print(f"Indices shape: {indices.shape}")
    print(f"topk={topk}, dim={D}, tail_dim={D_tail}")

    # Run kernel
    kernel_out, kernel_lse = sparse_mla_fwd_interface(
        q,
        kv,
        indices,
        sm_scale=None,
        d_v=D,
        block_I=block_I,
    )
    print(f"Kernel output shape: {kernel_out.shape}")
    print(f"Kernel LSE shape: {kernel_lse.shape}")

    # Run reference
    ref_out, ref_lse = sparse_mla_fwd_reference(
        q,
        kv,
        indices,
        dim=D,
        tail_dim=D_tail,
        kv_group=kv_group,
        sm_scale=None,
    )
    print(f"Reference output shape: {ref_out.shape}")
    print(f"Reference LSE shape: {ref_lse.shape}")

    # Compare
    torch.testing.assert_close(kernel_out, ref_out, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(kernel_lse, ref_lse, rtol=1e-2, atol=1e-1)
    print("\033[92mAll checks passed! Kernel matches reference.\033[0m")


if __name__ == "__main__":
    run_test()
