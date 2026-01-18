import torch
from typing import Tuple

from .tuner import jit_tuner
from .utils import get_num_sms, ceil_div, get_col_major_tma_aligned_tensor, get_m_alignment_for_contiguous_layout

# C++ code templates
includes = ('"deep_gemm/fp8_gemm.cuh"', )
template = """
using namespace deep_gemm;

// Templated args from Python JIT call
constexpr auto N = {N}, K = {K};
constexpr auto BLOCK_M = {BLOCK_M};
constexpr auto BLOCK_N = {BLOCK_N};
constexpr auto kNumStages = {NUM_STAGES};
constexpr auto kNumTMAMulticast = {NUM_TMA_MULTICAST};

// Make a templated GEMM
using GemmType = Gemm<N, K, BLOCK_M, BLOCK_N, 128, 1, kNumStages, kNumTMAMulticast, GemmType::Normal>;

// Launch kernel
auto tma_a_desc = GemmType::make_2d_tma_a_desc(lhs, m);
auto tma_b_desc = GemmType::make_2d_tma_b_desc(rhs);
auto tma_scales_a_desc = GemmType::make_2d_tma_scales_a_desc(lhs_scales, m);
auto tma_d_desc = GemmType::make_2d_tma_d_desc(out, m);
GemmType::run(out, rhs_scales, nullptr,
              m,
              tma_a_desc, tma_b_desc, tma_scales_a_desc, tma_d_desc,
              stream, num_sms, smem_size);
"""


def is_tma_multicast_legal(n: int, block_n: int, num_tma_multicast: int, num_sms: int) -> bool:
    if num_tma_multicast == 1:
        return True
    return (n % (block_n * num_tma_multicast) == 0) and num_sms % num_tma_multicast == 0


def get_smem_size(num_stages: int, k: int, block_m: int, block_n: int, block_k: int = 128) -> int:
    smem_d = block_m * block_n * 2
    smem_a_per_stage = block_m * block_k
    smem_scales_a_per_stage = block_m * 4
    smem_b_per_stage = block_n * block_k
    smem_scales_b = ceil_div(k, block_k) * 4
    smem_barrier = num_stages * 8 * 2

    smem_size = 0
    smem_size += smem_d
    smem_size += num_stages * smem_a_per_stage
    smem_size += num_stages * smem_scales_a_per_stage
    smem_size += num_stages * smem_b_per_stage
    smem_size += ceil_div(smem_scales_b * (1 if block_k % block_n == 0 else 2), 8) * 8
    smem_size += smem_barrier
    return smem_size


def get_best_configs(m: int, n: int, k: int, num_groups: int, num_sms: int,
                     is_grouped_contiguous: bool = False) -> Tuple[int, int, int, int, int, int]:
    #计算block_ms， 因为此例子中m=4096, 因此block_ms=128
    if not is_grouped_contiguous:
        # TODO: for some cases, smaller M block is better, add them into tuning space
        block_ms = (64 if m <= 64 else 128, )
    else:
        block_ms = (get_m_alignment_for_contiguous_layout(), )
    block_ns = tuple(range(16, 129, 8)) + (160, )

    """
    说一下我自己对这里算block_ns的理解，，block_ms和block_ns就是C矩阵中block tile的大小
    block_ms在非grouped_contiguous情况下固定为128。下面的代码是计算最优block_ns大小的代码
    计算block_ns的规则是这样的，首先h100有132个block，而block_ns的大小会影响最终block的数量
    代码中会对block_ns从16遍历到128，判断哪个block_ns的配置最优，最优的条件如下(也可以参考这个链接https://zhuanlan.zhihu.com/p/32383172703)：
    1、round(block_num / 132)的数量最少，在代码中将这个值称为wave
    2、在不同的block_ns情况的条件1的wave结果相同时，则根据block_num / 132的余数较大的情况选择block_ns的值
    然后说一下我对为什么要设置条件1和条件2的理解（这里的理解可能需要我把整个代码都看完了才能来写了，to be continued）
    还有关于num stage如何设置，以及代码中涉及到tma multicast相关的内容，需要等到我把代码完全看一遍了才能来写这里的理解了
    """
    #wave 理解为需要处理多少轮次， h800有132个streaming multi-processor, 
    #不同的block_ns意味者C矩阵子块个数不同，因此需要的轮次也就会不同
    #get_last_wave_util 则是在最后一轮中剩余的待处理子块，理论上c子块划分个数能整除132是最好的
    #如果不能整除，则在轮次一致的情况下，最后一轮待处理子块个数越多越好，因为此时计算可以分散在尽可能多的sm硬件资源上
    fix_wave_saturate = lambda x: num_sms if x == 0 else x
    get_num_waves = lambda bm, bn: (ceil_div(ceil_div(m, bm) * ceil_div(n, bn) * num_groups, num_sms) if bm else None)
    get_last_wave_util = lambda bm, bn: fix_wave_saturate((ceil_div(m, bm) * ceil_div(n, bn) * num_groups) % num_sms)

    # block_ns取值从16到128，步长8增长
    # 判断条件为轮次（get_num_waves）最少，在轮次一致的情况下，最后一轮待处理子块个数（get_last_wave_util）越多越好
    # 在本例中 block_ns=128
    # Decide block sizes by waves
    best_block_m, best_block_n = None, None
    for block_m in block_ms:
        for block_n in block_ns:
            success = False
            num_waves, best_num_waves = get_num_waves(block_m, block_n), get_num_waves(best_block_m, best_block_n)
            if best_block_m is None or best_block_n is None:
                success = True
            elif num_waves < best_num_waves:
                success = True
            elif num_waves == best_num_waves:
                # Check last wave utilization
                util = get_last_wave_util(block_m, block_n)
                best_util = get_last_wave_util(best_block_m, best_block_n)
                success = util > best_util or (util == best_util and (block_m > best_block_m or (block_m == best_block_m and block_n < best_block_n)))
            best_block_m, best_block_n = (block_m, block_n) if success else (best_block_m, best_block_n)
    assert best_block_m is not None and best_block_n is not None

    # 计算合适的stage, stage的判断条件为共享存储不超过232448的情况stage尽可能多
    # 因为stage越多，占用的共享存储越多
    # Always pick the longest one
    # NOTES: for double B scales, the best number of stages may be reduced
    best_num_stages, best_smem_size, sm90_capacity = None, None, 232448
    for num_stages in (6, 5, 4) if 128 % best_block_n != 0 else (8, 7, 6, 5, 4):
        best_smem_size = get_smem_size(num_stages, k, best_block_m, best_block_n)
        if best_smem_size <= sm90_capacity:
            best_num_stages = num_stages
            break
    assert best_num_stages is not None

    # Decide the number of TMA multicast
    best_num_tma_multicast = 1
    # When using large block tiling, broadcasting B is required to achieve maximum performance gains.
    if m >= 1024 and is_tma_multicast_legal(n, best_block_n, 2, num_sms) and num_groups == 1:
        best_num_tma_multicast = 2

    # Recompute the minimal number of SMs required
    # NOTES: less L2 cache usage and less GPU frequency drop
    num_waves = get_num_waves(best_block_m, best_block_n)
    num_min_sms = ceil_div(ceil_div(m, best_block_m) * ceil_div(n, best_block_n) * num_groups, num_waves)
    num_min_sms = ceil_div(max(num_min_sms, num_sms - 8), best_num_tma_multicast) * best_num_tma_multicast
    assert num_min_sms <= num_sms and is_tma_multicast_legal(n, best_block_n, best_num_tma_multicast, num_min_sms)

    return num_min_sms, best_block_m, best_block_n, best_num_stages, best_num_tma_multicast, best_smem_size


def gemm_fp8_fp8_bf16_nt(lhs: Tuple[torch.Tensor, torch.Tensor],
                         rhs: Tuple[torch.Tensor, torch.Tensor],
                         out: torch.Tensor) -> None:
    """
    Do a normal GEMM with FP8 inputs and BF16 output, with 1x128 LHS scaling and 128x128 RHS scaling.
    LHS, RHS, RHS scaling factors, and output tensors must be in contiguous format.
    RHS and RHS scaling factors are required to be transposed.
    The LHS scaling tensor requires TMA-aligned transposed format, if your input does not match the requirement,
        this function will do a transposing with a set of slow PyTorch operations.

    Arguments:
        lhs: the first element is an FP8 tensor (typed `torch.float8_e4m3fn`) of shape `[m, k]`,
             the second element is an FP32 1x128 scaling tensor for LHS of shape `[m, ⌈k / 128⌉]`.
        rhs: the first element is an FP8 tensor (typed `torch.float8_e4m3fn`) of shape `[n, k]`.
             the second element is an FP32 128x128 scaling tensor for RHS of shape `[⌈n / 128⌉, ⌈k / 128⌉]`.
        out: the BF16 output tensor of shape `[m, n]`, representing the result.

    执行一个标准的 GEMM 运算，输入为 FP8 格式，输出为 BF16 格式，使用 1x128 的 LHS 缩放和 128x128 的 RHS 缩放。
    LHS（左侧矩阵）、RHS（右侧矩阵）、RHS 缩放因子和输出张量必须是连续内存格式。
    RHS 和 RHS缩放因子 需要被转置。（在construct里面构造RHS和对应缩放因子的时候就是n*k的转置形式了，都是行主序的）
    LHS缩放张量 需要 TMA 对齐的转置格式，如果你的输入不满足这个要求，该函数会使用一组较慢的 PyTorch 操作来进行转置。

    参数：
        lhs：第一个元素是形状为 [m, k] 的 FP8 张量（类型为 torch.float8_e4m3fn），
            第二个元素是形状为 [m, ceil(k / 128)] 的 FP32 1x128 缩放张量，用于 LHS。
        rhs：第一个元素是形状为 [n, k] 的 FP8 张量（类型为 torch.float8_e4m3fn）。
            第二个元素是形状为 [ceil(n / 128), ceil(k / 128)] 的 FP32 128x128 缩放张量，用于 RHS。
        out：形状为 [m, n] 的 BF16 输出张量，表示计算结果。
    """
    lhs, lhs_scales = lhs
    rhs, rhs_scales = rhs
    m, k = lhs.shape
    n, k_ = rhs.shape
    m_, n_ = out.shape

    assert n % 64 == 0 and k % 128 == 0

    # Type and shape checks
    assert m == m_ and n == n_ and k == k_
    assert n > 0 and k > 0
    assert lhs_scales.shape == (m, (k + 127) // 128)
    assert rhs_scales.shape == ((n + 127) // 128, (k + 127) // 128)
    assert lhs.dtype == torch.float8_e4m3fn and lhs_scales.dtype == torch.float32
    assert rhs.dtype == torch.float8_e4m3fn and rhs_scales.dtype == torch.float32
    assert out.dtype == torch.bfloat16
    assert lhs.is_contiguous() and rhs.is_contiguous() and out.is_contiguous()

    # LHS scales must be transposed for TMA load, but not for RHS scales
    # NOTES: `get_tma_aligned_lhs_scales` may launch a kernel if not processed by previous kernels
    lhs_scales = get_col_major_tma_aligned_tensor(lhs_scales)
    assert rhs_scales.is_contiguous()

    # Do nothing if `m` is zero
    if m == 0:
        return

    # Auto-tuning with compilation
    global includes, template
    # 这里返回gpu的sm个数
    num_sms = get_num_sms()
    num_sms, block_m, block_n, num_stages, num_tma_multicast, smem_size = get_best_configs(m, n, k, 1, num_sms)
    args = (lhs, lhs_scales, rhs, rhs_scales, out, m, torch.cuda.current_stream(), num_sms, smem_size)
    # 注意，这里jit编译后进行调用的地方是fp8_geeemm.cuh中的Gemm::run方法，这个文件最上面的宏展开可以说明一切
    runtime = jit_tuner.compile_and_tune(
        name='gemm_fp8_fp8_bf16_nt',
        keys={'N': n, 'K': k, 'BLOCK_M': block_m, 'BLOCK_N': block_n,
              'NUM_STAGES': num_stages, 'NUM_TMA_MULTICAST': num_tma_multicast},
        space=(),
        includes=includes,
        arg_defs=(('lhs', torch.float8_e4m3fn), ('lhs_scales', torch.float),
                  ('rhs', torch.float8_e4m3fn), ('rhs_scales', torch.float),
                  ('out', torch.bfloat16), ('m', int),
                  ('stream', torch.cuda.Stream), ('num_sms', int), ('smem_size', int)),
        template=template,
        args=args
    )

    # Run the kernel
    runtime(*args)
