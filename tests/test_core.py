import random
import torch
from typing import Tuple

import deep_gemm
from deep_gemm import bench_kineto, calc_diff, ceil_div, get_col_major_tma_aligned_tensor
# import sys
# sys.path.append('/home/p/Workspace/code/cuda_learn/DeepGEMM')
# from deep_gemm.utils import bench_kineto, calc_diff
# from deep_gemm.jit_kernels.utils import ceil_div, get_col_major_tma_aligned_tensor


def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # 这里使用了e4m3的fp8的对称量化，这里x的shape为4096*7168（m*k）
    # 还有注意这里函数说是做的per token量化，但这里的per token量化和我直接理解的per token量化感觉不太一样
    # 之前看论文里面讲的per token量化是，假设一个矩阵是num_tokens * hidden_size，那么per token量化就是对每个token的hidden_size这个vector做量化，hidden_size维度使用一个scale
    # 但这里的per token量化明显不是这样的, 这里的fp8量化更加的细粒度。这里的per token量化是，这里的x的shape是4096*7168（我理解可以看作4096个token，每个token的hidden dim是7168）
    # 然后将x view为4096*56*128，相当于每个token的vector变为了56份，每份是一个128的vector，fp8量化的时候则是128长度的vector单独使用一个scale
    # 这样就相当于每个token实际包含了56个scale，整个x矩阵则包含了4096*56个scale
    # 可以看到代码中return的结果，第一个结果是量化过后的x矩阵，shape被还原到了4096*7168，第二个结果是scale矩阵，shape是4096*56
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    # 这里448是e4m3的最大值，对应的二进制为01111110， 2^8 * 1.75=448
    return (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), (x_amax / 448.0).view(m, -1)


def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # 这里也是用的fp8对称量化，e4m3的格式，不过量化的方式很特殊，和我之前学过的都不一样
    # 这里输入x的shape是2112x7168（n*k），在处理的时候会被padding到2176x7168（因为2112不能被128整除，所以会padding到2176）
    # 代码中会将2176x7168的矩阵分为128x128个block（在代码中对应x_view，shape为17x128x56x128），总共有17x56个block
    # 然后对每个block中的数据单独计算scale，并进行量化
    # 最后的返回结果有两个，一个是经过量化后的x矩阵(shape为2112x7168)，另一个是scale矩阵(shape为17x56)
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros((ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(x_view.size(0), x_view.size(2))


def construct(m: int, k: int, n: int) -> \
        Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
    out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    ref_out = x @ y.t()

    x_fp8, y_fp8 = per_token_cast_to_fp8(x), per_block_cast_to_fp8(y)
    # Transpose earlier so that the testing will not trigger transposing kernels
    # 说一下这里调用get_col_major_tma_aligned_tensor的作用，主要作用就是将x_fp8[1]从行主序改为列主序
    # x_fp8[1]的shape为4096*56，stride原本为(56, 1)，这表示行主序
    # 在执行了这个函数之后，shape仍然为4096*56，但stride变成了(1, 4096)，这表示列主序
    # 这个函数还有一个作用就是保证tma的16byte对齐要求，这个我还没深入研究，这个后面有时间了再说
    x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
    return x_fp8, y_fp8, out, ref_out


def construct_grouped(num_groups: int, m: int, k: int, n: int, is_masked: bool) -> \
        Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    x = torch.randn((num_groups, m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)
    out = torch.empty((num_groups, m, n), device='cuda', dtype=torch.bfloat16)
    ref_out = torch.einsum('gmk,gnk->gmn', x, y)

    assert m % 4 == 0, f'TMA alignment error: {m}'
    x_fp8 = (torch.empty_like(x, dtype=torch.float8_e4m3fn), torch.empty((num_groups, m, k // 128), device='cuda', dtype=torch.float))
    y_fp8 = (torch.empty_like(y, dtype=torch.float8_e4m3fn), torch.empty((num_groups, (n + 127) // 128, k // 128), device='cuda', dtype=torch.float))
    for i in range(num_groups):
        x_fp8[0][i], x_fp8[1][i] = per_token_cast_to_fp8(x[i])
        y_fp8[0][i], y_fp8[1][i] = per_block_cast_to_fp8(y[i])

    # For non-masked input, we must merge the group and M dims
    if not is_masked:
        x_fp8 = (x_fp8[0].view(-1, k), per_token_cast_to_fp8(x.view(-1, k))[1])
        out, ref_out = out.view(-1, n), ref_out.view(-1, n)

    # Transpose earlier so that the testing will not trigger transposing kernels
    x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
    return x_fp8, y_fp8, out, ref_out


def test_gemm() -> None:
    print('Testing GEMM:')
    # 后面的讲解就以m=4096, k=7168, n=2112为例
    for m in (64, 128, 4096):
        for k, n in [(7168, 2112), (1536, 24576), (512, 32768), (16384, 7168), (7168, 4096), (2048, 7168)]:
            x_fp8, y_fp8, out, ref_out = construct(m, k, n)
            deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)
            diff = calc_diff(out, ref_out)
            assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'

            # Construct new tensors only once to avoid L2 cache acceleration (creating them puts them in L2)
            x_fp8, y_fp8, out, ref_out = construct(m, k, n)

            # noinspection PyShadowingNames
            def test_func():
                deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)

            t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
            print(f' > Performance (m={m:5}, n={n:5}, k={k:5}): {t * 1e6:4.0f} us | '
                  f'throughput: {2 * m * n * k / t / 1e12:4.0f} TFLOPS, '
                  f'{(m * k + k * n + m * n * 2) / 1e9 / t:4.0f} GB/s')
    print()


def test_m_grouped_gemm_contiguous() -> None:
    print('Testing grouped contiguous GEMM:')

    for num_groups, m, k, n in ((4, 8192, 7168, 4096), (4, 8192, 2048, 7168), (8, 4096, 7168, 4096), (8, 4096, 2048, 7168)):
        # TODO: make a stronger test
        x_fp8, y_fp8, out, ref_out = construct_grouped(num_groups, m, k, n, is_masked=False)
        m_indices = torch.arange(0, num_groups, device='cuda', dtype=torch.int)
        m_indices = m_indices.unsqueeze(-1).expand(num_groups, m).contiguous().view(-1)
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(x_fp8, y_fp8, out, m_indices)
        diff = calc_diff(out, ref_out)
        assert diff < 0.001, f'm={m * num_groups}, {k=}, {n=}, {diff:.5f}'

        # Construct new tensors only once to avoid L2 cache acceleration (creating them puts them in L2)
        x_fp8, y_fp8, out, ref_out = construct_grouped(num_groups, m, k, n, is_masked=False)
        m_indices = torch.arange(0, num_groups, device='cuda', dtype=torch.int)
        m_indices = m_indices.unsqueeze(-1).expand(num_groups, m).contiguous().view(-1)

        # noinspection PyShadowingNames
        def test_func():
            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(x_fp8, y_fp8, out, m_indices)

        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        print(f' > Performance ({num_groups=}, m_per_group={m:4}, n={n:4}, k={k:4}): {t * 1e6:4.0f} us | '
              f'throughput: {2 * num_groups * m * n * k / t / 1e12:4.0f} TFLOPS, '
              f'{(num_groups * (m * k + k * n + m * n * 2)) / 1e9 / t:4.0f} GB/s')
    print()


def test_m_grouped_gemm_masked() -> None:
    print('Testing grouped masked GEMM:')

    for num_groups, m in ((1, 1024), (2, 512), (4, 256)):
        for k, n in ((7168, 4096), (2048, 7168), ):
            # Test correctness
            masked_m_candidates = list(filter(lambda candidate: candidate <= m, (64, 128, 192, 256, 320, 384)))
            for i in range(10):
                x_fp8, y_fp8, out, ref_out = construct_grouped(num_groups, m, k, n, is_masked=True)
                masked_m = torch.empty((num_groups, ), device='cuda', dtype=torch.int)
                for j in range(num_groups):
                    masked_m[j] = random.choice(masked_m_candidates)
                expected_m = min(int(masked_m.float().mean()) + 1, m)
                deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(x_fp8, y_fp8, out, masked_m, expected_m)
                for j in range(num_groups):
                    diff = calc_diff(out[j, :masked_m[j].item()], ref_out[j, :masked_m[j].item()])
                    assert diff < 0.001, f'{m=}, {k=}, {n=}, {j=}, masked_m={masked_m[j]}, {num_groups=}, {diff:.5f}'

            # Construct new tensors only once to avoid L2 cache acceleration (creating them puts them in L2)
            x_fp8, y_fp8, out, ref_out = construct_grouped(num_groups, m, k, n, is_masked=True)
            masked_m = torch.ones((num_groups, ), device='cuda', dtype=torch.int) * m

            # noinspection PyShadowingNames
            def test_func():
                deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(x_fp8, y_fp8, out, masked_m, m)

            # Test performance with fixed shapes
            t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
            print(f' > Performance ({num_groups=}, m_per_group={m:4}, n={n:4}, k={k:4}): {t * 1e6:4.0f} us | '
                  f'throughput: {2 * num_groups * m * n * k / t / 1e12:4.0f} TFLOPS, '
                  f'{(num_groups * (m * k + k * n + m * n * 2)) / 1e9 / t:4.0f} GB/s')
    print()


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    test_gemm()
    test_m_grouped_gemm_contiguous()
    test_m_grouped_gemm_masked()
    
    # construct(4096, 7168, 2112)
