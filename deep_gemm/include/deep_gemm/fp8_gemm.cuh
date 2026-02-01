#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"
#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include "mma_utils.cuh"
#include "scheduler.cuh"
#include "tma_utils.cuh"
#include "utils.cuh"

namespace deep_gemm {

enum class Layout {
    RowMajor,
    ColMajor
};

template <uint32_t kNumTMAThreads, uint32_t kNumMathThreadsPerGroup>
__device__ __host__ constexpr int get_num_threads_per_sm(int block_m) {
    DG_STATIC_ASSERT(kNumMathThreadsPerGroup == 128, "Only support 128 threads per math group");
    return (block_m == 64 ? 1 : 2) * kNumMathThreadsPerGroup + kNumTMAThreads;
}

template <uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumGroups, uint32_t kNumStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreadsPerGroup,
          uint32_t kNumTMAMulticast,
          GemmType kGemmType>
__global__ void __launch_bounds__(get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M), 1)
fp8_gemm_kernel(__nv_bfloat16* gmem_d, float* scales_b, int* grouped_layout,
                uint32_t shape_m,
                const __grid_constant__ CUtensorMap tensor_map_a,
                const __grid_constant__ CUtensorMap tensor_map_b,
                const __grid_constant__ CUtensorMap tensor_map_scales_a,
                const __grid_constant__ CUtensorMap tensor_map_d) {
// #if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)
    // Scaling checks
    DG_STATIC_ASSERT(BLOCK_K == 128, "Only support per-128-channel FP8 scaling");
    DG_STATIC_ASSERT(ceil_div(BLOCK_N, BLOCK_K) == 1 or (gcd(BLOCK_N, BLOCK_K) == BLOCK_N - BLOCK_K), "Too much B scales in a single block");
    
    // 本例使用SM90_64x160x32_F32E4M3E4M3_SS的wgmma
    // Types
    using WGMMA = typename FP8MMASelector<BLOCK_N>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    // Shared memory
    static constexpr int kMustUseUniformedScaleB = (BLOCK_K % BLOCK_N == 0);
    static constexpr uint32_t SMEM_D_SIZE = BLOCK_M * BLOCK_N * sizeof(__nv_bfloat16);
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_SCALES_A_SIZE_PER_STAGE = BLOCK_M * sizeof(float);
    static constexpr uint32_t SHAPE_K_SCALES = ceil_div(SHAPE_K, BLOCK_K);
    static constexpr uint32_t SMEM_SCALES_B_SIZE = ceil_div<uint32_t>(SHAPE_K_SCALES * (kMustUseUniformedScaleB ? 1 : 2) * sizeof(float), sizeof(Barrier)) * sizeof(Barrier);

    // Configs
    constexpr uint32_t kFullKOfAllStages = kNumStages * BLOCK_K; // block_k默认128，num stage本例为5
    constexpr uint32_t kNumThreads = get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M); // 128 * 2 + 128 = 384
    constexpr uint32_t kNumMathThreads = kNumThreads - kNumTMAThreads; // 256
    constexpr uint32_t kNumIterations = ceil_div(SHAPE_K, kFullKOfAllStages); // 7168 / (5 * 128) = 12
    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = get_lane_id();

    // Prefetch TMA descriptors at very beginning
    // 参考reed的博客https://zhuanlan.zhihu.com/p/1985678344352731952
    // 这里是对tensormap做预取，即博客里面讲的第三种获取tensormap的方法
    // 在第三种方法中，tensormap放在hbm上，TMA单元从HBM经由L2 Cache来获得tensormap
    // 注意这种prefetch是一种从gmem中拿数据的操作，会有一定延迟
    // 还有就是TMA单元为了提升操作效率会有TMA Descriptor的Cache机构
    // 所以tma在做tensormap预取时会优先查cache，所有就可能出现cache hit和cache miss
    if (threadIdx.x == kNumMathThreads) {
        // lhs的tensormap
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_a));
        // rhs的tensormap
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_b));
        // lhs scale的tensormap
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_scales_a));
        // out的tensormap
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_d));
    }
    // 这里只对当前warp中的线程做同步, __syncthreads()则会对整个block做同步
    __syncwarp();

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_D_SIZE % 1024 == 0, "Shared memory of A/B must be aligned to 1024 bytes");

    // Data on shared memory
    auto smem_d = reinterpret_cast<__nv_bfloat16*>(smem_buffer);
    __nv_fp8_e4m3* smem_a[kNumStages];
    __nv_fp8_e4m3* smem_b[kNumStages];
    float* smem_scales_a[kNumStages];
    float* smem_scales_b;

    // TMA Barrier for both divisible and non-divisible cases
    Barrier* full_barriers[kNumStages];
    Barrier* empty_barriers[kNumStages];

    // 这里的逻辑好理解，就是先预留smem空间，然后把对应的指针塞到vector里面
    // Fill shared memory pointers
    #pragma unroll
    for (int i = 0; i < kNumStages; ++ i) {
        smem_a[i] = reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_D_SIZE + i * SMEM_A_SIZE_PER_STAGE);
        smem_b[i] = reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_D_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
        smem_scales_a[i] = reinterpret_cast<float*>(smem_buffer + SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE) + i * SMEM_SCALES_A_SIZE_PER_STAGE);
    }
    smem_scales_b = reinterpret_cast<float*>(smem_buffer + SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_SCALES_A_SIZE_PER_STAGE));

    // 这里应该是在计算mbarrier的起始地址，需要注意的是，mbarrier是存储在smem上的
    // Fill barriers
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(reinterpret_cast<uint8_t*>(smem_scales_b) + SMEM_SCALES_B_SIZE);
    /*
    说一下这里，在本例中K=7168，kNumStages=5，BLOCK_K=128
    kNumIterations = ceil_div(7168, 5 * 128) = 12
    这里的full_barriers的作用是生产者通知消费者tma的数据已经准备好了，empty_barriers的作用是消费者通知生产者数据消费完了
    这里full_barriers和empty_barriers数组的长度都为5，表示一共有5个stage，每个stage都有自己单独的barrier
    你也可以看到在代码中，不管是tma生产数据，还是wgmma消费数据，都是按stage来生产和消费数据的（都是最内层的for循环生产/消费数据）
    还有这里的for循环中，每次都是+i，你可能问，一个mbarrier不是8byte吗，为什么这里+1不是+8呢？
    ai说是因为这里的full_barriers和empty_barriers都是Barrier*类型的指针数组，所以+i实际上是+i*sizeof(Barrier)，也就是+i*8byte
    */
    #pragma unroll
    for (int i = 0; i < kNumStages; ++ i) {
        full_barriers[i] = barrier_start_ptr + i;
        empty_barriers[i] = barrier_start_ptr + kNumStages + i;
    }

    // Initialize barriers
    DG_STATIC_ASSERT(kNumTMAMulticast <= 32, "Too many TMA multicast");
    if (threadIdx.x == kNumMathThreads) {
        /*
        这里参考reed大佬的mbarrier博客https://zhuanlan.zhihu.com/p/1962636004235153810
        每个mbarrier大致可以分为4个主要字段、phase、arrive count、expected arrive count、transaction count(tx)
        其中phase字段用于标记barrier的状态，只能为0或者1，0和1本身没有啥含义，具体含义根据代码中的逻辑判断
        phase字段的作用和ampere的多级流水gemm中的reg_store_idx、reg_load_idx、smem_sel（leetcuda代码中的变量）这类的0 1变量的作用非常类似，只是表示当前所处阶段的状态
        arrive count需要和expected arrive count一起来看。以如下代码为例，empty_barriers[i]->init(16)表示将empty_barriers[i]中
        的arrive count和expected arrive count都初始化为-16，block中只要有一个线程调用了empty_barriers[i]->arrive()，那么arrive count就会加1，这里
        需要注意的是，由于mbarrier是存放在smem上的，所以block中所有线程都可以执行empty_barriers[i]->arrive()操作，只要有一个thread执行了empty_barriers[i]->arrive()，那么
        arrive count就会+1，而expected arrive count只是用来记录arrive count最开始的初始值，所以不管怎么调用arrive，expected arrive count的值一直会是-16保持不变
        当arrive count增加到0位置，phase就回反转（从0变到1或者从1变到0），那么代码中执行empty_barriers[i]->wait(xxphase)的地方就回从阻塞状态变为非阻塞状态
        需要注意的是，当phase反转之后，arrive count会重新变为expected arrive count的初始值（即-16）

        还有就是transaction count（tx）字段，这个一般和tma一起用，这里以full_barriers为例，代码中将full_barriers[i]初始化为1
        full_barriers[i]的arrive count和expected arrive count都为-1，此时tx被初始化为0
        可以通过full_barriers[i]->expect_transaction(xx)来设置tx的初始值为-xx(单位是byte)，则mbarrier中实际的tx值为-xx。
        这里需要full_barriers[i]的arrive count和tx同时到达0，phase才会反转，full_barriers[i]->wait(xxphase)才会从阻塞状态变为非阻塞状态。
        关于如何让arrive count和tx的值从负数变为0，对于arrive count，还是线程调用arrive方法让arrive count自增
        而对于tx的数值的增加，则是将full_barriers[i]传递给tma，当tma完成aa byte数据拷贝时，tx的值就会增加aa
        当tx和arrive count都变为0之后，phase就会反转，full_barriers[i]->wait(xxphase)才会从阻塞状态变为非阻塞状态，同时arrive count又重新变为expected arrive count的值，而tx还是保持0
        还有就是，可以看到tma的while循环中使用了full_barrier.arrive_and_expect_tx(xx)方法，这个方法的作用是，让arrive count自增+1，同时设置tx从0变为-xx，具体的可以参考reed的博客

        还有就是代码中empty_barriers[i]可以看到初始化的arrive count是乘以了mulicast值的，参考https://github.com/deepseek-ai/DeepGEMM/issues/51这个链接
        reed大佬的这里https://zhuanlan.zhihu.com/p/1985678344352731952也说过，“TMA还能提供Cluster内的Multicast能力，即读取一份数据可以组播给cluster内的多个block”
        当M大于64时，kNumTMAMulticast=2，也就是说一个cluster包含两个block，一次tma会拷贝两份数据到两个block中，我理解是因为这个所以empty_barriers[i]可以看到初始化的arrive count是乘以了mulicast值
        还有就是注意在consumer中当empty_barriers[i]使用arrive时，arrive需要传入cta的id作为参数(0和1)，consumer代码中有这块的逻辑
        */
        // NOTES: we always use `lane_idx` to arrive for the `lane_idx`-th CTA in the cluster,
        // even with TMA multicast disabled, we want to make the behavior aligned
        // 注意：我们始终使用 `lane_idx` 来针对cluster中的第 `lane_idx` 个 CTA 执行到达（arrive）操作，即便是禁用了 TMA multicast，我们也希望保持行为一致
        #pragma unroll
        for (int i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(kNumTMAMulticast * kNumMathThreads / 32);
        }
        
        // 这两行代码暂时不清楚干什么的
        // 主要是这两行代码即使没搞清楚，也不影响后续代码的理解阅读
        // 这两行代码的作用先放着，后续有空了再看是做什么的
        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_view_async_shared();
        (kNumTMAMulticast > 1) ? cutlass::arch::fence_barrier_init() : void();
    }

    // Synchronize all threads to make barrier visible in normal memory model
    (kNumTMAMulticast > 1) ? cute::cluster_sync() : __syncthreads();

    // For pipeline unrolling
    struct DivisibleK {};
    struct NotDivisibleK {};
    /*
    这里解释一下，这里的launch_k_iterations是一个lambda函数，同时while循环里面的launch_k_iterations也是一个lambda函数
    但这两个lambda函数有一些区别。总的来说这两个lambda函数共同作用，完成了在编译器去做循环展开的功能
    下面把两个lambda函数根据代码中出现的顺序简称为第一个lambda和第二个lambda
    第一个lambda描述用于在SHAPE_K维度做循环展开，代码中将num stage个block_k合并的称为一个iter，在本例中，一个iter对应了5*128的k维度大小
    在tma的while循环之中，最外层的循环就是iter的循环，一次iter就处理5*128的k维度，内层循环是num stage（本例中为5）的循环，一次循环处理一个stage(128的k)
    而第一个lambda就是定义了iter循环的循环展开的骨架，使得编译器能在编译器对iter循环做循环展开
    第二个lambda函数中描述了业务逻辑，描述的是stage循环和其中的tma逻辑
    第二个lambda中的逻辑相当于第一个lambda中的func函数，第一个lambda在调用func函数(第二个lambda)时，会传入两个参数，k_iter和DivisibleK{} / NotDivisibleK{}，这两个参数在func函数的定义中都是会被用到的
    展开后的代码示例如下：
    // 迭代 0 (普通)
    func(0, DivisibleK{}); 
    // 迭代 1 (普通)
    func(1, DivisibleK{}); 
    // 迭代 2 (尾部，最后一轮)
    func(2, NotDivisibleK{});

    这里通过lambda函数对iter循环做循环展开有这么几个好处：
    1、编译器可以根据传入的不同shape自己做循环展开，不用人为手动展开
    2、第一个lambda做展开后，可以消除掉第一个lambda中的if else判断开销，也能消除掉iter的for循环开销
    3、由于展开是编译期做的，外部的iter展开的数量是在编译期就能确定的，所以内部的stage循环的#pragma unroll也能被编译器给展开
    */
    auto launch_k_iterations = [](const auto& func) {
        // 这里if else判断的目的是，如果SHAPE_K(7168)能整除kFullKOfAllStages(5 * 128)，则执行if分支，否则执行else分支
        // 在本例中，7168 / (5 * 128) = 11.2，不能被整除，也就是说最后一个一个k_iter(kNumIterations在这里为12，向上取整)需要特殊处理，所以走的else分支
        if constexpr (SHAPE_K % kFullKOfAllStages == 0) {
            for (int k_iter = 0; k_iter < kNumIterations; ++ k_iter)
                func(k_iter, DivisibleK{});
        } else {
            for (int k_iter = 0; k_iter < kNumIterations - 1; ++ k_iter)
                func(k_iter, DivisibleK{});
            func(kNumIterations - 1, NotDivisibleK{});
        }
    };

    // Register reconfigurations
    constexpr int kNumTMARegisters = 40;
    constexpr int kNumMathRegisters = 232;

    // 初始化scheduler，scheduler用于决定当前sm执行完当前block任务之后，下一步需要执行哪个block
    // Block scheduler
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = Scheduler<kGemmType, SHAPE_N, BLOCK_M, BLOCK_N, kNumGroups, kNumTMAMulticast>(shape_m, grouped_layout); // 普通gemm的grouped_layout为nullptr

    if (threadIdx.x >= kNumMathThreads) {
        // 这个指令直接看ptx文档就行，讲的蛮详细的
        // 作用就是，表示限制当前所有执行该setmaxnreg.dec的warp中的thread、每个thread的寄存器数量减少到kNumTMARegisters个
        // 减少之后每个thread多出来的寄存器则会归还到寄存器pool
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

        // 这里注意一下，这个分支是tma的分支，一共有128个thread执行到这个分支
        // 但经过下面的if判断之后，实际上128个thread中只有一个thread会进行tma操作
        // 即只block中的第kNumMathThreads（256）线程参与数据加载
        // NOTES: only one thread (or warp) will be used
        if (threadIdx.x == kNumMathThreads) {
            // 这里的作用是得到要处理的block的index
            // 注意这里是while循环，因为使用了持久化内核，一个sm处理完了一个block之后会继续处理下一个block
            // 直到sm把它需要处理的block都计算完之后，才会退出
            // Persistently schedule over blocks
            while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
                launch_k_iterations([&](int k_iter, auto type) {
                    // 这里的type就是第一个lambda函数传入的DivisibleK{} / NotDivisibleK{}
                    // 这里是判断传入的type和DivisibleK是否一致
                    constexpr bool kHasDivisibleStages = std::is_same_v<decltype(type), DivisibleK>;
                    constexpr int kNumInnerStages = kHasDivisibleStages ? kNumStages : (SHAPE_K % kFullKOfAllStages) / BLOCK_K;
                    DG_STATIC_ASSERT(kNumInnerStages != 0, "Invalid number of inner stages");

                    // NOTES: unrolling and `kNumInnerStages` are vital for performance, NVCC will try to eliminate all
                    // shared memory pointers, e.g. `full_barriers` registers, if all the access indices are constant
                    // 注意：循环展开和 `kNumInnerStages` 对性能至关重要，如果所有访问索引均为常量，NVCC 将尝试消除所有共享内存指针，例如 `full_barriers` 寄存器
                    #pragma unroll
                    for (uint32_t s = 0; s < kNumInnerStages; ++ s) {
                        // 等待consumer将empty barriers中的数据消费完，然后对应的phase反转
                        // 这里的wait是阻塞操作，直到phase反转才会继续向下执行
                        // 还有就是这里的wait的参数，这个参数实际就是一个0 / 1值，即wait参数中的值为一个期望的phase值，这里设期望的phase值为xx
                        // 如果当前barrier的phase值和wait期望的phase值不一致，wait就会一直阻塞，否则就向下执行
                        // 当consumer消费完数据时，mbarrier的phase反转，反转过后值若为xx，则这里的wait就可以解除阻塞，继续向下执行
                        // 我觉得这里wait中的那个复杂的参数变量为什么这么写我还没研究
                        // Wait consumer release
                        empty_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter + 1) & 1);

                        // Issue TMA A with broadcasting
                        auto& full_barrier = *full_barriers[s];
                        // 这里应该是算的当前block处于整体的k维度的哪个位置
                        // k_iter * kFullKOfAllStages就是iter level的便宜
                        // s * BLOCK_K就是stage level的偏移
                        int k_idx = k_iter * kFullKOfAllStages + s * BLOCK_K;
                        // 这里用tma载入lhs和lhs scale矩阵，需要注意的是这里的tma使用了multicast
                        // 之所以使用multicast，我理解是因为，是因为在deepgemm的block启动策略中，相邻的block处理的数据处于同一行，只是列不同
                        // 也就是说 ，相邻block处理的A矩阵的数据是一样的，所以可以用tma的multicast，发起一次数据拷贝，将数据放到两个block中
                        // lhs矩阵一次载入shape为128*128个数，lhs scale矩阵一次载入为128*1
                        // 待解决的点：假设有两个相邻的block刚好需要换行，那这两个block处理的A矩阵就不是同一行了，这个时候还能用mulicast吗
                        tma_copy<kNumTMAMulticast>(&tensor_map_a, reinterpret_cast<uint64_t*>(&full_barrier),
                                                   smem_a[s], k_idx, scheduler.get_global_idx(shape_m, BLOCK_M, m_block_idx));
                        tma_copy<kNumTMAMulticast>(&tensor_map_scales_a, reinterpret_cast<uint64_t*>(&full_barrier),
                                                   smem_scales_a[s], m_block_idx * BLOCK_M,
                                                   scheduler.get_global_idx(SHAPE_K_SCALES, 1, k_idx / BLOCK_K));

                        // 如上面所描述，由于相邻两个block处理的rhs矩阵是不同列的，数据不相同，所以不能用multicast
                        // rhs矩阵一次载入为160*128个数
                        // Issue TMA B without broadcasting
                        tma_copy(&tensor_map_b, reinterpret_cast<uint64_t*>(&full_barrier),
                                 smem_b[s], k_idx, scheduler.get_global_idx<false>(SHAPE_N, BLOCK_N, n_block_idx, m_block_idx));
                        // 这里调用arrive_and_expect_tx方法，该方法的作用是，将arrive count自增加1，同时设置tx为负的SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_SCALES_A_SIZE_PER_STAGE
                        // 也就是说从执行了这个命令开始，arrive count就为0了，但tx为负的SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_SCALES_A_SIZE_PER_STAGE
                        // 需要等到tma全部拷贝完了，tx自增到0了之后，full_barrier的phase才会反转
                        full_barrier.arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_SCALES_A_SIZE_PER_STAGE);
                    }
                    // 这里貌似是为了防止死锁，具体的情况是当最后一个iter的stage个数不为5的时候会进行这里的逻辑
                    // 更具体的我没深究
                    // Wait unaligned cases
                    #pragma unroll
                    for (uint32_t s = kNumInnerStages; s < kNumStages; ++ s) {
                        empty_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter + 1) & 1);
                        full_barriers[s]->arrive();
                    }
                });
            }
            // 这里是持久化内核处理完成时，针对multicast情况，退出时的一些额外处理
            // 这块我没仔细研究
            // To safely deconstruct distributed shared barriers, we need another round of empty waits
            if constexpr (kNumTMAMulticast > 1) {
                #pragma unroll
                for (uint32_t s = 0; s < kNumStages; ++ s)
                    empty_barriers[s]->wait((scheduler.current_iter * kNumIterations + 1) & 1);
            }
        }
    } else {
        // Math warp-groups for WGMMA
        // 限制当前所有执行该setmaxnreg.inc的warp中的thread、每个thread的寄存器数量增加到kNumMathRegisters个，
        // 每个thread会向寄存器pool申请额外的寄存器以使得寄存器数量达到kNumMathRegisters个
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();
        // 这里kNumMathThreadsPerGroup固定为128，表示一个warpgroup（4个warp）
        // mathwarp一共包含256个线程，也就是两个warpgroup
        // 这里的math_wg_idx表示当前线程处于第几个warpgroup当中
        // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
        const auto math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / kNumMathThreadsPerGroup, 0);
        // 关于这两个变量的含义，放到下面用到r0和r1的地方讲
        const auto r_0 = warp_idx * 16 + lane_idx / 4, r_1 = r_0 + 8;

        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {

            /*
            下面这一坨代码，解释一下是干什么的
            下面这里什么num_former_iters，num_full_iters，最后都是为计算num_scales_b服务的
            对于num_scales_b的计算公式，是SHAPE_K_SCALES乘上一个 1或者2 的系数
            在这里SHAPE_K_SCALES为7168 / 128 = 56
            对于 1或者2 的系数具体取什么值，通过num_former_iters >= num_full_iters来判断
            我这里直接说结论。对于本例中，BLOCK_N = 160，大于了rhs量化的n的粒度（rhs量化n的粒度为128，rhs中128*128(n*k)大小的矩阵是一个量化单位）
            由于rhs中128*128(n*k)大小的矩阵是一个量化单位，而这里BLOCK_N = 160大于了128，也就是说rhs的shape 160*128跨越了两个128*128的区域
            所以需要加载两个128*128矩阵的量化scale，所以num_scales_b = SHAPE_K_SCALES * 2 = 112
            说一下这里num_scales_b的含义，表示当前block总共需要载入多少个scales_b的值
            本例中rhs的shape是2112*7168(n*k)，rhs scales的shape是17*56，量化的粒度是，rhs中128*128的矩阵对应一个scale
            在mathwarp的双层for循环中，总共需要载入160*7168(block_n*K)的rhs矩阵数据，才能完成结果矩阵中128*160(block_m*block_n)的结果计算
            而block_n由于大于了128，跨越了两个per block量化的区域，所以在n维度上需要载入两个scale b
            同时由于需要载入160*7168的rhs矩阵做计算，7168在k维度上等于56个block_k的大小，所以k维度上需要载入56个scale
            所以总共需要载入2*56 = 112，num_scales_b的含义就是这样来的
            也就是说，对于一个block的计算，rhs scale矩阵需要载入112个float32值

            这里说一下num_former_iters和num_full_iters表示反量化的迭代次数，主要用在下面的promotion反量化步骤中
            这里由于block_n=160大于了per block量化的n为128的量化粒度，所以每个block tile在n维度会跨越两个量化block
            (这里注意区分block tile和量化block，block tile指的的结果矩阵的block大小，shape为128*160。而量化block指的是对rhs矩阵做量化时的量化block大小，大小为128*128)
            所以在载入rhs scale时，需要载入两个量化scale（就是两个float）。载入量化scale之后，还需要进行反量化步骤，
            而反量化步骤是在for循环中进行的，每次循环执行一个寄存器组的反量化（关于这里寄存器组的概念，见下面promotion部分的注释）
            在promotion的反量化部分，for循环执行20次，这20次反量化步骤会将block tile的n维度跨越了0到160
            所以这就涉及到，这20次的循环中，哪部分循环使用第一个rhs量化scale，哪部份循环使用第二个量化scale
            这里的num_former_iters就是for循环中使用第一个量化scale的循环步数
            num_full_iters为总的需要的循环迭代数，在本例中num_full_iters一般算出来都为20，和WGMMA::kNumAccum/4的结果相等
            但当触及到矩阵边界是，num_full_iters会小于20
            */
            // Decide the number of scales B to load
            DG_STATIC_ASSERT(SHAPE_N % 8 == 0, "Invalid shape N");
            uint32_t num_former_iters = BLOCK_N / 8, num_full_iters = num_former_iters;
            // kMustUseUniformedScaleB: 判断block k能否整除block n，若不能整除则执行则执行if中的逻辑
            // 这里的block k为128，block n为160，按理来说是会执行这里的if分支的
            if constexpr (not kMustUseUniformedScaleB) {
                num_former_iters = min(BLOCK_N, BLOCK_K - n_block_idx * BLOCK_N % BLOCK_K) / 8;
                num_full_iters = min(SHAPE_N - n_block_idx * BLOCK_N, BLOCK_N) / 8;
            }
            // 在本例中SHAPE_K_SCALES在这里为 7168 / 128 = 56
            // num_scales_b为112
            uint32_t num_scales_b = SHAPE_K_SCALES * (num_former_iters >= num_full_iters ? 1 : 2);
            /*
            这里的代码需要分情况讨论
            首先对于普通fp8 gemm的情况，这里num_previous_lines恒定为0
            因此，变量local_scales_b表达式中可以吧表达式中的num_previous_lines忽略
            所以local_scales_b = scales_b + ((n_block_idx * BLOCK_N) / BLOCK_K) * SHAPE_K_SCALES，然后来解释一下这个表达式的含义
            首先scales_b为rhs scales矩阵的gmem地址
            rhs scale矩阵的shape为17*56(n*k)。由于kernel中对于rhs的k维度做遍历，所以rhs scales的k维度是会被block全部载入的
            也就是说kernel中需要载入rhs_scales[row,:]，所以在kernel中需要确定载入rhs scales矩阵中的哪一部分数据的变量就是row
            而这里的(n_block_idx * BLOCK_N) / 128，告诉kernel需要载入17*56的那一行的数据
            n_block_idx表示当前block在结果矩阵中的n维度index，n_block_idx * BLOCK_N就表示表示在结果矩阵中的n维度的偏移
            (n_block_idx * BLOCK_N) / 128就表示当前属于17*56的17中的第几个(你别忘了17*56中的17，每一行都代表了一个n维度的128的rhs量化的数据)
            所以local_scales_b = rhs_scale的gmem地址 + rhs_scales矩阵的行偏移 * rhs_scales矩阵的列数(也就是56)

            然后就是for循环部分，首先for循环只会让mathwarp[32:255]执行，mathrwarp[0:32](第一个warp)不会执行这里的载入rhs_scales的任务
            这里注释也说了，是想让第一个warp的tma写回操作和这里的rhs scales load操作 overlap起来
            这里代码中，在一个block结果算完之后，会使用第一个warp的第一个线程执行tma操作将数据写回
            然后剩下mathwarp[32:255]的线程就会进入下一个循环，然后mathwarp[32:255]就会执行rhs scales load操作
            这样tma写回和rhs scales load就overlap了起来，而不需要这两个访存操作完全串行的操作
            然后说一下这里for循环部分的执行逻辑。首先for循环写清楚之后是这样
            for (uint32_t i = threadIdx.x - 32; i < 112; i += 224)
            即实际上，在mathwarp中，能满足这个循环条件的从而进入循环体的，只有mathwarp[32:144)这112个线程能进入循环体执行
            同时由于for循环一次之后i就会自增224，从而不满足i < 112的条件而退出
            所以mathwarp[32:144)这些线程在for循环中只能执行一次
            然后我们再来看st_shared(smem_scales_b + i, __ldg(local_scales_b + i))做了什么
            这里smem_scales_b + i中，smem_scales_b是rhs scale的起始地址，i就是偏移量
            local_scales_b是需要读取的rhs scales中的起始地址，i就是偏移量
            所以这里的命令实际上就是将gmem中local_scales_b + i这个位置的数据读到smem中这个位置
            for循环执行完成之后，当前block需要的112个scale值就完全被载入smem中
            这里需要说一下，这里载入是112个线程参与载入的，每个线程实际只读了1个float32
            然后说一下st_shared和__ldg的作用
            __ldg参考红皮书，__ldg()中传入gmem的地址。作用是针对只读数据（整个kernel运行期间不能向这里写数据），从gmem中载入的时候通过“read only data cache”进行缓存(其实是纹理内存)。纹理缓存是l1上的cache
            st_shared对应了st.shared.f32的ptx，查阅了ptx文档之后，说这个指令的作用是将数据存放到指定的smem地址上

            还有就是说一下为什么这里关于rhs scale，不使用tma加载, 而使用st.shared.f32加载。以下是个人观点
            rhs scale矩阵全部加起来也就17*56大小，而这里一个warpgroup实际只需要其中两个元素
            由于需要的数据量特别小，所以没必要用tma加载，直接用st.shared.f32就行
            */
            // Load B scales with math warp-groups
            // NOTES: except the first warp, we want to overlap loading B scales with TMA stores between tasks
            if (threadIdx.x >= 32) {
                auto num_previous_lines = scheduler.get_global_idx<false>(ceil_div(SHAPE_N, BLOCK_K), 0, 0, m_block_idx);
                auto local_scales_b = scales_b + (num_previous_lines + ((n_block_idx * BLOCK_N) / BLOCK_K)) * SHAPE_K_SCALES;
                #pragma unroll
                for (uint32_t i = threadIdx.x - 32; i < num_scales_b; i += kNumMathThreads - 32)
                    st_shared(smem_scales_b + i, __ldg(local_scales_b + i));
            }

            
            // 这行代码的作用是，同步256个math thread，而不同步剩余的128个tma线程
            cutlass::arch::NamedBarrier(kNumMathThreads).sync();

            // WGMMA::kNumAccum为80
            // Accumulation for WGMMA or CUDA promotion
            float accum[WGMMA::kNumAccum], final_accum[WGMMA::kNumAccum] = {0};

            // lambda函数，用于empty_barriers的arrive操作
            // Empty barrier arrival
            auto empty_barrier_arrive = [&](int s) {
                if constexpr (kNumTMAMulticast == 1) {
                    lane_idx == 0 ? empty_barriers[s]->arrive() : void();
                } else {
                    lane_idx < kNumTMAMulticast ? empty_barriers[s]->arrive(lane_idx) : void();
                }
            };

            // 和tma那里同样的使用lambda函数在编译器做循环展开
            // Launch MMAs
            launch_k_iterations([&](int k_iter, auto type) {
                constexpr bool kHasDivisibleStages = std::is_same_v<decltype(type), DivisibleK>;
                constexpr int kNumInnerStages = kHasDivisibleStages ? kNumStages : (SHAPE_K % kFullKOfAllStages) / BLOCK_K;
                DG_STATIC_ASSERT(kNumInnerStages != 0, "Invalid number of inner stages");

                #pragma unroll
                for (int s = 0; s < kNumInnerStages; ++ s) {
                    // 这里应该是将smem_scales_b + k_iter * kNumStages + s位置的smem数据载入到scale_b_0寄存器
                    // smem_scales_b是scales_b的smem起始地址，k_iter * kNumStages表示iter层面的偏移，s是stage层面的偏移
                    // 拿出来的一个float32的scale数据代表了一个k维度上128的scale
                    // Read B scales
                    float scale_b_0 = ld_shared(smem_scales_b + k_iter * kNumStages + s), scale_b_1;
                    // 这里因为block_n大于block_k，所以rhs矩阵跨越了两个量化单位区域，所以需要载入第二个scale_b
                    // scale_b_0和scale_b_1代表rhs矩阵中两个128*128块的量化scale，这两个块在n维度上相邻的，而k维度则相同
                    // NOTES: even some blocks do not need to read the second row, but we still load one to align with other blocks
                    if constexpr (not kMustUseUniformedScaleB)
                        scale_b_1 = ld_shared(smem_scales_b + k_iter * kNumStages + s + SHAPE_K_SCALES);

                    // 等待tma对应stage的full_barriers的phase的变化
                    // Wait TMA arrivals
                    full_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter) & 1);

                    /*
                    说一下这里怎么从smem_scales_a中拿到需要的两个scale数据的
                    下面讲scale_a_0和scale_a_1简称为scale0和scale1
                    首先，在本例中，smem_scales_a[s]表示第s个stage的lhs scale的起始地址
                    每个stage的lhs scale的shape为128*1，128表示128行
                    这里需要说明一下lhs scale整体的shape估计你才能理解上面这一行注释是什么意思
                    首先，本例中lhs矩阵大小为4096*7168(M*K)，lhs矩阵是per group量化
                    所以lhs scale矩阵的大小为4096*56，对于lhs scale矩阵，56中的每一列都代表了K维度的128列
                    然后这里每个block由于要在k维度做slice，所以iter循环加上内部的stage循环，总共需要128*56的lhs scale，对应的lhs矩阵的shape为128*7168
                    但上面tma在载入lhs和lhs scale时，是按照stage做载入的，对于lhs，每次tma载入128*128的内容；对应的tma载入lhs scale，stage循环中tma一次载入128*1(这里的1对应了k维度的128列)的lhs scale
                    所以这里smem_scales_a[s]所表示的数据长度为128，0～127的index就是表示的第几行(M维度)的scale
                    所以这里r_0和r_1起始是代表的行偏移，即当前线程负责那几行的数据的反量化计算，所以需要取两行的scale
                    (这里的r_0和r_1的含义和我之前写的余弦相似度里面的每个线程里面用于记录对应行的平方和的寄存器非常的类似，也是每个线程负责存储不同行的数据)
                    然后是为什么每个线程需要取两行的scale呢，这个需要你去看看ptx文档中关于wgmma m64Nk32的数据结果在warpgroup不同线程的数据分布
                    在分布图中，每个线程会存放64*N(本例中N为160)的结果矩阵中的两行的结果
                    以warpgroup的0号线程为例，0号线程的结果寄存器存储了64*160结果的第0行和第8行的部分数据
                    所以第0号线程在对0号线程中的结果寄存器中结果做反量化时，需要0行和8行的scale，所以这里需要两个寄存器scale_a_0和scale_a_1
                    而这里的r_0和r_1就是线程id在“m64Nk32的数据结果在warpgroup不同线程的数据分布图”中对应的行index
                    (r_0 = warp_idx * 16 + lane_idx / 4, r_1 = r_0 + 8)
                    ld_shared就是传入smem指针，把指针位置的数据载入到寄存器中
                    
                    还有就是关于下面NOTES注释那里的含义没弄明白
                    */
                    // Read A scales
                    // 注意：必须在调用 `warpgroup_arrive` 之前完成所有共享内存读取操作，
                    // 这是为了防止下一个被调度执行的 Block Tile 覆盖共享内存中的数据，导致当前块的计算结果被污染
                    // NOTES: all shared memory read must be prior to `warpgroup_arrive` to avoid next scheduled block polluting the results
                    auto scale_a_0 = ld_shared(smem_scales_a[s] + r_0), scale_a_1 = ld_shared(smem_scales_a[s] + r_1);

                    // 关于这里warpgroup_fence_operand指令，在wgmma算完之前和commit之后都被调用了
                    // warpgroup_fence_operand的具体作用，参考这里https://github.com/NVIDIA/cutlass/discussions/1375
                    // 作用解释是“这不会对内核产生任何正确性影响。它仅仅是一个NVVM代码移动屏障，用于确保在WGMMA指令批处理过程中，其他任何操作都不会触及WGMMA指令的寄存器。”
                    // Commit WGMMA instructions
                    #pragma unroll
                    for (int i = 0; i < WGMMA::kNumAccum; ++ i)
                        warpgroup_fence_operand(accum[i]);
                    
                    /*
                    这里统一讲一下这里的几个wgmma用到的指令
                    首先说明一下，由于reed还没有写wgmma的博客，同时我又自己懒得看ptx，所以这里关于这里用到的wgmma指令只讲个大概，后面完全看懂了再来重新补充
                    首先是warpgroup_arrive()指令，参考这个https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/（翻译链接https://mp.weixin.qq.com/s/ysvE4PBiKkljwFfBQAN1Jw）的说法是，
                    ptx中是这样解释的(上面链接中也有类似的话术)“wgmma.fence instruction must be used to fence the register accesses of wgmma.mma_async instruction from their prior accesses. Otherwise, the behavior is undefined.”
                    翻译过来是“必须使用 wgmma.fence指令(就是这里的warpgroup_arrive())对 wgmma.mma_async指令的寄存器访问与其先前访问进行隔离。否则，行为将未定义。”
                    意思应该是说，在当前wgmma使用相关寄存器进行使用之前，需要加上warpgroup_arrive，保证其他对该寄存器的操作执行完了
                    未完持续：但这里有个问题，下面不是已经有了warpgroup_wait<0>()来保证wgmma算完了才继续执行吗，为什么这里又要加上warpgroup_arrive()
                    然后就是创建a和b的tensor的descriptor，然后算wgmma，这块也可以参考for循环中的那一大串注释
                    然后就是warpgroup_commit_batch()和warpgroup_wait<0>()，这两个参考的这里https://zhuanlan.zhihu.com/p/32383172703
                    warpgroup_commit_batch()的作用是将创建的还未提交的wgmma.mma_async提交到当前warpgroup做执行
                    (这里额外说一嘴，一些链接和ptx文档提到过，在warpgroup_arrive和wgmma.mma_async之间有时需要插入fence.proxy.async指令保证结果正确性，但这个我还没研究)
                    wgmma.wait_group.sync.aligned N 执行的含义是(这里抄的ptx文档的，正确性不用质疑)：将使执行线程等待，直到最近的 wgmma-group中仅有 N个或更少处于待处理状态，且该线程之前提交的所有 wgmma-group均已完成。例如：当 N=0时，执行线程将等待所有先前的 wgmma-group完成。操作数 N为整型常量
                    还有就是关于这里的mma和promotion的overlap的问题，参考这4个链接https://mp.weixin.qq.com/s/ub4tgxeAK-YlQniUSDD3zA https://mp.weixin.qq.com/s/O-YHkpFCVC9Tch6QcLwoHw https://github.com/deepseek-ai/DeepGEMM/issues/152 https://github.com/NVIDIA/cutlass/issues/218github.com/NVIDIA/cutlass/issues/2181
                    首先是deepgemm这里没有在不同的计算warpgroup之间做barrier同步，是属于cooperative的写法。这种写法两个计算warpgroup的执行之间没有同步，两个计算warpgroup之间会争抢tensorcore资源，overlap完全来自warp scheduler的调度
                    两个warpgroup之间的执行进度差距不会很大，不会出现一个warpgroup领先另一个warpgroup很多的情况
                    所以这里的mma和promotion在两个计算warpgroup之间的overlap，可以理解为，确实存在一定的overlap，但不会overlap的很完美，且有时候可能会退化到完全不overlap的情况
                    */
                    warpgroup_arrive();
                    // 本例使用SM90_64x160x32_F32E4M3E4M3_SS的wgmma
                    // WGMMA::M WGMMA::N WGMMA::K分别为64 160 32
                    // 这里的wgmma对smem中的数据做计算，结果写到reg中(wgmma也可以对reg中的数据做计算，结果写到reg中)
                    #pragma unroll
                    for (int k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                        // 这里有两个warp group（每个warp group有128个线程）
                        // 参考这里https://zhuanlan.zhihu.com/p/1946940443301483665 (此时此刻由于reed大佬关于hopper的只写到了tma，还没写到wgmma，所以关于wgmma只能暂时参考一下其他人写的博客了，而且我也懒得去看ptx文档。。)，即
                        // wgmma可以对smem中的数据做计算，也可以对reg中的数据做计算
                        // 如果是对reg中的数据做计算，那么数据的分布就和mma类似，每个thread的reg中放一点数据
                        // 如果是对smem中的数据做计算，那么需要给wgmma传入smem的描述符，描述附通过make_smem_desc
                        // 这个smem描述符，看起来是一个warp group中的所有线程都是一样的，而不像mma那种每个thread中的数据都不一样
                        // 这个desc_a和desc_b看起来，对于每一个warp group内的所有线程，他们的desc_a和desc_b算出来是一样的
                        // 我理解这个desc_a和desc_b是warp group级别描述smem中tensor的描述符
                        // 这里我们需要计算128*160*32的矩阵乘，然后for循环4次（就是上面的那个k循环），
                        // 然后进行累加，就得到了128*160*128的矩阵乘结果。然后由于这里有两个warp group，所以每一个warp group都是使用一个64x160x32的wgmma来计算
                        // 两个warp group加起来刚好就是计算了一个128*160*32的结果。然后上面也提到了，一个warp group内所有线程的desc_a和desc_b是相同的
                        // 对于desc_a，第一个参数传入参数含义是，当前处理的slice k数据(64*32大小)在smem_a中的起始地址
                        // 第一个参数中，smem_a[s]就是第s个stage的A矩阵的起始地址（每个stage的A矩阵的size为128*128 block_m*block_k）
                        // math_wg_idx * WGMMA::M * BLOCK_K的含义是，首先math_wg_idx表示当前是第一个warp group，WGMMA::M表示每个wgmma的m维度大小，在这里为64，block_k就是128
                        // 所以math_wg_idx * WGMMA::M * BLOCK_K表示的是warp group level的行偏移（ 0/1 * 64 * 128 ）
                        // k * WGMMA::K的含义是，k就是第几个小k的坐标，WGMMA::K就是32，所以k * WGMMA::K就是slice k的偏移（列偏移）
                        // 然后desc_a的make_smem_desc的第二个参数表示使用的swizzle的部署模式，这个定义在cutlass/include/cute/arch/mma_sm90_desc.hpp第48行开始的那个枚举类中，这里传入1表示
                        // 使用B128类型的swizzle。由此看出wgmma将swizzle直接封装到了硬件当中（回想上面的tma操作，一次tma也是载入一个stage的数据，对于A矩阵来说，一次tma载入128*128的数据 block_m*block_k）
                        // 而这里desc_b，同样的，make_smem_desc中第一个传入参数为当前slice数据(64*32大小)在smem_b中的起始地址
                        // 可以看到desc_b相比desc_a的传入参数，少了warp group level的行偏移，只保留了slice k的列偏移
                        // 这是因为这里maththread有两个warp group，这两个warp group在做wgmma计算的时候，只对M维度做了拆分（每个warp group计算64的M，合起来就是128）
                        // 而在N维度，2个warp group之间不用做切分，都是算的N=160，所以desc_b这里没有warp group level的行偏移。desc_b的第二个参数同理，也是表示用怎样的swizzle
                        // 然后就是wgmma的写入寄存器accum相关的内容。这里accum的定义为float accum[WGMMA::kNumAccum]，其中WGMMA::kNumAccum在这里为80
                        // 也就是说每个线程的accum有80个float32的寄存器。首先注意一下，wgmma使用float32的累加器，即这里的accum
                        // 然后就是为什么是每个线程有80个float32 reg呢，这个其实好理解，这里计算wgmma的M N为64*160，燃油一个warp group中有128个线程
                        // 128*80 = 64*160，所以一个warp group中，每个线程有80个float32寄存器，就能把M N为64*160的wgmma的计算结果装下
                        auto desc_a = make_smem_desc(smem_a[s] + math_wg_idx * WGMMA::M * BLOCK_K + k * WGMMA::K, 1);
                        auto desc_b = make_smem_desc(smem_b[s] + k * WGMMA::K, 1);
                        WGMMA::wgmma(desc_a, desc_b, accum, k);
                    }
                    warpgroup_commit_batch();
                    // WGMMA::kNumAccum为80
                    #pragma unroll
                    for (int i = 0; i < WGMMA::kNumAccum; ++ i)
                        // 参考https://github.com/NVIDIA/cutlass/discussions/1375
                        warpgroup_fence_operand(accum[i]);
                    warpgroup_wait<0>();

                    /*
                    wgmma消费完成后，将对应stage的empty_barriers的arrive count从-16自增到0
                    这里需要额外说一下，因为这里是multicast情况（multicast为2）
                    同时math线程有256个，总共8个warp
                    这里调用的是这样的代码lane_idx < kNumTMAMulticast ? empty_barriers[s]->arrive(lane_idx) : void()
                    可以看到，一个warp中实际上有两个线程、0和1号线程，都调用了arrive方法，且arrive方法竟然把lane_idx作为了参数
                    8个warp中，每个warp调用两遍arrive，刚好能把arrive count从-16自增到0
                    还有就是，这里的arrive(lane_idx)中接受的参数实际上是cta_idx，即cta的下标
                    这里multicast为2，刚好对应了cta 0和1
                    */
                    // Notify barrier arrival
                    empty_barrier_arrive(s);

                    /*
                    这里的for循环中的内容。首先是i循环的次数，在本例中，由于WGMMA::kNumAccum / 4 = 20，所以for循环为20次
                    至于为什么需要20次。这个需要根据m64Nk32的ptx输出分布图来看，在本例中N为160，所以是m64n160k32
                    在分布图中，对于每个线程的寄存器，以T0线程为例，设4个float32的寄存器为一组，例如图中的{d0,d1,d2,d3}为第0组，{d4,d5,d6,d7}为第1组
                    这里的d0 d1到dn其实就对应了本代码中的accum[80]，都是float32类型
                    在ptx分布图中，{d0,d1,d2,d3}一组寄存器对应了N=160中的第0列和第一列的结果，{d4,d5,d6,d7}对应了第8列和第9列的结果，这两组寄存器之间的列偏移为8
                    也就是说，这里block_n为160，每组寄存器之间的列偏移为8，所以需要以8为单位做反量化，所以你可以看到num_former_iters和num_full_iters在计算时会除以8
                    160 / 8 = 20，就对应了这里的WGMMA::kNumAccum / 4 = 20。总共需要循环20次，每一次循环在N维度上反量化两列的结果
                    然后就是这里accum[i * 4 + 0/1/2/3]中的i * 4 + 0/1/2/3的含义
                    首先i就是表示第几组寄存器，每个线程80个寄存器(accum[80])按照4为一组被分为了20组，每组4个寄存器
                    这里的i表示第几组寄存器，i*4表示寄存器组间的偏移量，+ 0/1/2/3表示组内的偏移量
                    关于num_former_iters的含义，写在这个变量定义处的注释中
                    还有就是这里的英文注释说使用predicate对性能很重要。问了下ai，说是这样的原因：
                    如果不使用predicate的话，就需要用两个循环来实现这样的逻辑，代码如下。第一个for循环由于
                    num_former_iters不是编译期常量，第一个for循环实际没办法展开
                        #pragma unroll
                        for (int i = 0; i < num_former_iters; ++ i) {
                            final_accum[i * 4 + 0] += scale_0_0 * accum[i * 4 + 0];
                            final_accum[i * 4 + 1] += scale_0_0 * accum[i * 4 + 1];
                            final_accum[i * 4 + 2] += scale_1_0 * accum[i * 4 + 2];
                            final_accum[i * 4 + 3] += scale_1_0 * accum[i * 4 + 3];
                        }

                        // 后半部分使用 scale_b_1
                        #pragma unroll
                        for (int i = num_former_iters; i < WGMMA::kNumAccum / 4; ++ i) {
                            final_accum[i * 4 + 0] += scale_0_1 * accum[i * 4 + 0];
                            final_accum[i * 4 + 1] += scale_0_1 * accum[i * 4 + 1];
                            final_accum[i * 4 + 2] += scale_1_1 * accum[i * 4 + 2];
                            final_accum[i * 4 + 3] += scale_1_1 * accum[i * 4 + 3];
                        }
                    而如果使用predicate的话，只需要一个for循环，同时for循环的值是编译期常量，可以循环展开
                    同时bool predicate = kMustUseUniformedScaleB or i < num_former_iters这句话可以被编译为SELECT指令，不会引入分支，不会导致流水线停顿
                    */
                    // Promote with scales
                    // 使用谓词（predicate）的方式对性能非常重要，相比于使用两个独立的循环
                    // NOTES: making it as predicates is very important for performance, comparing to two loops
                    float scale_0_0 = scale_a_0 * scale_b_0, scale_1_0 = scale_a_1 * scale_b_0;
                    float scale_0_1, scale_1_1;
                    if constexpr (not kMustUseUniformedScaleB)
                        scale_0_1 = scale_a_0 * scale_b_1, scale_1_1 = scale_a_1 * scale_b_1;
                    #pragma unroll
                    for (int i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                        bool predicate = kMustUseUniformedScaleB or i < num_former_iters;
                        final_accum[i * 4 + 0] += (predicate ? scale_0_0 : scale_0_1) * accum[i * 4 + 0];
                        final_accum[i * 4 + 1] += (predicate ? scale_0_0 : scale_0_1) * accum[i * 4 + 1];
                        final_accum[i * 4 + 2] += (predicate ? scale_1_0 : scale_1_1) * accum[i * 4 + 2];
                        final_accum[i * 4 + 3] += (predicate ? scale_1_0 : scale_1_1) * accum[i * 4 + 3];
                    }
                }
                // 这里应该是边界条件操作，具体的情况是当最后一个iter的stage个数不为5的时候会进行这里的逻辑
                // 这么操作是为了解决什么，我没有深究
                // Wait unaligned cases
                #pragma unroll
                for (uint32_t s = kNumInnerStages; s < kNumStages; ++ s) {
                    full_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter) & 1);
                    empty_barrier_arrive(s);
                }
            });

            // 这里是把final_accum中的反量化的fp32数据转换成bf16类型，然后用stmatrix从reg写到smem
            // __float22bfloat162_rn的作用是将两个fp32数据转换成一个bf162数据(一个bf162包含两个bf16数据)
            // Write back to shared memory using STSM
            DG_STATIC_ASSERT(WGMMA::kNumAccum % 4 == 0, "Invalid STSM x2 vectorization");
            #pragma unroll
            for (auto i = 0; i < WGMMA::kNumAccum / 8; ++ i) {
                SM90_U32x4_STSM_N<nv_bfloat162>::copy(
                    __float22bfloat162_rn({final_accum[i * 8 + 0], final_accum[i * 8 + 1]}),
                    __float22bfloat162_rn({final_accum[i * 8 + 2], final_accum[i * 8 + 3]}),
                    __float22bfloat162_rn({final_accum[i * 8 + 4], final_accum[i * 8 + 5]}),
                    __float22bfloat162_rn({final_accum[i * 8 + 6], final_accum[i * 8 + 7]}),
                    smem_d + (warp_idx * 16 + lane_idx % 16) * BLOCK_N + i * 16 + 8 * (lane_idx / 16)
                );
            }
            if constexpr (WGMMA::kNumAccum % 8 != 0) {
                SM90_U32x2_STSM_N<nv_bfloat162>::copy(
                    __float22bfloat162_rn({final_accum[WGMMA::kNumAccum / 8 * 8 + 0], final_accum[WGMMA::kNumAccum / 8 * 8 + 1]}),
                    __float22bfloat162_rn({final_accum[WGMMA::kNumAccum / 8 * 8 + 2], final_accum[WGMMA::kNumAccum / 8 * 8 + 3]}),
                    smem_d + (warp_idx * 16 + lane_idx % 16) * BLOCK_N + WGMMA::kNumAccum / 8 * 16
                );
            }
            /*
            tma的数据写回, smem->gmem
            对于tma的写回，使用的是commit和wait来管理的，而不是像tma读入那样的mbarrier管理
            需要注意的是，这里的tma写回和计算的else分支最开始的load b scale是overlap起来的
            使用第1个warpgroup的第0好线程执行tma写回
            */
            // 调用的fence.proxy.async.shared::cta，该指令的作用还没研究
            cute::tma_store_fence();
            // 这行代码的作用是，同步256个math thread，而不同步剩余的128个tma线程
            cutlass::arch::NamedBarrier(kNumMathThreads).sync();

            // Use TMA store to write back to global memory
            if (threadIdx.x == 0) {
                cute::SM90_TMA_STORE_2D::copy(&tensor_map_d, smem_d, n_block_idx * BLOCK_N,
                                              scheduler.get_global_idx(shape_m, BLOCK_M, m_block_idx));
                //调用的cp.async.bulk.commit_group指令
                cute::tma_store_arrive();
                cute::tma_store_wait<0>();
            }
            __syncwarp();
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");
#endif
}

template <uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumGroups, uint32_t kNumStages,
          uint32_t kNumTMAMulticast,
          GemmType kGemmType>
class Gemm {
private:
    using Barrier = cuda::barrier<cuda::thread_scope_block>;

public:
    Gemm() = default;

    static void run(__nv_bfloat16* gmem_d, float* scales_b, int* grouped_layout,
                    uint32_t shape_m,
                    const CUtensorMap& tma_a_desc,
                    const CUtensorMap& tma_b_desc,
                    const CUtensorMap& tma_scales_a_desc,
                    const CUtensorMap& tma_d_desc,
                    cudaStream_t stream,
                    int num_sms, uint32_t smem_size) {
        // 这里注意一个点，就是代码中128thread负责tma，但128个thread中实际只有一个thread是真正在做tma的
        // 所以只使用一个thread就能做tma，那为什么tma线程不分配一个warp呢，而要分配4个warp？
        // 关于这个问题，gemini 3pro这样回答(下面的英文注释也是相同的意思)：因为代码中使用了setmaxnreg指令去优化tma warp的reg，减少tma warp的reg使用，将多出来的reg分给wgmma warp，提高计算warp的occ
        // 而setmaxnreg指令要求操作对象最小的单位是4个warp，所以tma线程分配了4个warp
        // NOTES: we must use 4 warps to do TMA, because `setmaxnreg.aligned` requires 4 warps
        constexpr uint32_t kNumTMAThreads = 128;
        constexpr uint32_t kNumMathThreadsPerGroup = 128;
        auto kernel = fp8_gemm_kernel<SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N, BLOCK_K,
                                      kNumGroups, kNumStages, kNumTMAThreads, kNumMathThreadsPerGroup,
                                      kNumTMAMulticast, kGemmType>;
        DG_HOST_ASSERT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) == cudaSuccess);

        // Cluster launch
        cudaLaunchConfig_t config;
        /**
        !!!!!!!!!!!!
        持久化内核，只启动sm数量个block
        !!!!!!!!!!!!
        **/
        config.gridDim = num_sms;
        // 按照本例中m n k分别为4096 2112 7168的情况，BLOCK_M为128
        // 这个函数返回一个block中的线程数量，在本例中为128*2 + 128，其中128*2 thread负责计算，另外128 thread负责TMA
        config.blockDim = get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M);
        config.dynamicSmemBytes = smem_size;
        config.stream = stream;

        // Clusters for TMA multicast
        // NOTES: `>= 4` cluster size will cause performance degradation
        cudaLaunchAttribute attr;
        attr.id = cudaLaunchAttributeClusterDimension;
        // 参考https://zhuanlan.zhihu.com/p/708645371，这里是设置cluster size，这里为2
        attr.val.clusterDim = {kNumTMAMulticast, 1, 1};
        config.attrs = &attr;
        config.numAttrs = 1;

        // Launch
        // 参考https://zhuanlan.zhihu.com/p/708645371，使用dsmem时需要使用cudaLaunchKernelEx启动kernel
        auto status = cudaLaunchKernelEx(&config, kernel,
                                         gmem_d, scales_b, grouped_layout,
                                         shape_m,
                                         tma_a_desc, tma_b_desc, tma_scales_a_desc, tma_d_desc);
        DG_HOST_ASSERT(status == cudaSuccess);
    }

    /**
    参考https://zhuanlan.zhihu.com/p/1985678344352731952 https://zhuanlan.zhihu.com/p/709750258
    这里说明一下这里的TMA Descriptor的创建，TMA Descriptor通过cuTensorMapEncodeTiled方法创建，得到的是一个CUtensorMap对象
    构造tma descriptor的cuTensorMapEncodeTiled的传参如下所示
    CUresult cuTensorMapEncodeTiled (
            CUtensorMap* tensorMap,
            CUtensorMapDataType tensorDataType,
            cuuint32_t tensorRank,
            void* globalAddress,
            const cuuint64_t* globalDim,
            const cuuint64_t* globalStrides,
            const cuuint32_t* boxDim,
            const cuuint32_t* elementStrides,
            CUtensorMapInterleave interleave,
            CUtensorMapSwizzle swizzle,
            CUtensorMapL2promotion l2Promotion,
            CUtensorMapFloatOOBfill oobFill);
    以这里的make_2d_tma_a_desc(lhs矩阵)为例，deepgemm中在gemm.py最上面会通过auto tma_a_desc = GemmType::make_2d_tma_a_desc(lhs, m)来调用make_2d_tma_a_desc
    传入的lhs是torch的lhs的global addr，m就是行数。调用make_2d_tma_a_desc之后，会一层层往下调用，最终调用到cuTensorMapEncodeTiled的位置，此时
    传参关系是这样的，首先tensorMap就是一个空的CUtensorMap对象指针，tensorDataType就是e4m3，tensorRank在deepgemm的实现中都默认为2，表示二维矩阵，
    globalAddress就是lhs矩阵的global地址，globalDim在这里是[7168, 4096]（本例中m n k分别为4096 2112 7168），
    globalStrides在这里是7168 * sizeof(e4m3)，boxDim（也叫smem_dim）在这里是[128, 128] (代码默认写死BLOCK_K大小为128)，
    elementStrides为[1, 1]（代码中写死了elementStrides只能为[1, 1]）, 
    interleave写死了只能为CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE，
    swizzle为CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B

    关于其他矩阵的tma descriptor的创建，没时间写了，自己看代码
    **/
    template <typename T>
    static CUtensorMap make_2d_tma_a_desc(T* global_address, uint32_t shape_m) {
        return make_2d_tma_desc(global_address, Layout::RowMajor,
                                shape_m * (kGemmType == GemmType::GroupedMasked ? kNumGroups : 1), SHAPE_K, BLOCK_M, BLOCK_K);
    }

    template <typename T>
    static CUtensorMap make_2d_tma_b_desc(T* global_address) {
        return make_2d_tma_desc(global_address, Layout::ColMajor,
                                SHAPE_K, SHAPE_N * (kGemmType != GemmType::Normal ? kNumGroups : 1), BLOCK_K, BLOCK_N);
    }

    template <typename T>
    static CUtensorMap make_2d_tma_d_desc(T* global_address, uint32_t shape_m) {
        return make_2d_tma_desc(global_address, Layout::RowMajor,
                                shape_m * (kGemmType == GemmType::GroupedMasked ? kNumGroups : 1), SHAPE_N,
                                min(BLOCK_M, shape_m), BLOCK_N,
                                CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE);
    }

    template <typename T>
    static CUtensorMap make_2d_tma_scales_a_desc(T* global_address, uint32_t shape_m) {
        // Make TMA aligned to 16 bytes
        constexpr uint32_t kAlignment = 16 / sizeof(T);
        shape_m = ceil_div(shape_m, kAlignment) * kAlignment;

        return make_2d_tma_desc(global_address, Layout::ColMajor,
                                shape_m, ceil_div(SHAPE_K, BLOCK_K) * (kGemmType == GemmType::GroupedMasked ? kNumGroups : 1), BLOCK_M, 1,
                                CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE);
    }

    template <typename T>
    static CUtensorMap make_2d_tma_desc(
            T* global_address, Layout layout,
            uint32_t gmem_rows, uint32_t gmem_cols,
            uint32_t smem_rows, uint32_t smem_cols,
            CUtensorMapSwizzle swizzle_type = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B) {
        if (layout == Layout::RowMajor) {
            uint64_t gmem_dim[2] = {gmem_cols, gmem_rows};
            uint32_t smem_dim[2] = {smem_cols, smem_rows};
            return make_2d_tma_copy_desc(global_address, gmem_dim, gmem_cols * sizeof(T), smem_dim, swizzle_type);
        } else {
            uint64_t gmem_dim[2] = {gmem_rows, gmem_cols};
            uint32_t smem_dim[2] = {smem_rows, smem_cols};
            return make_2d_tma_copy_desc(global_address, gmem_dim, gmem_rows * sizeof(T), smem_dim, swizzle_type);
        }
    }
};

};  // namespace deep_gemm

#pragma clang diagnostic pop
