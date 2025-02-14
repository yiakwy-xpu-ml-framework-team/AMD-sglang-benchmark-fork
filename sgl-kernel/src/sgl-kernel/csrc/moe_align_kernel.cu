// Adapted from https://github.com/vllm-project/vllm/blob/v0.6.5/csrc/moe/moe_align_sum_kernels.cu

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <THC/THCAtomics.cuh>

#include "utils.hpp"

#ifdef USE_ROCM
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#else
#include <cooperative_groups.h>
#endif // USE_ROCM
#include <iostream> // TODO (yiakwy) : remove
#ifndef USE_ROCM
#define WARP_SIZE 32
#else
#define WARP_SIZE warpSize
#endif

#ifndef USE_ROCM
#define DevFuncAttribute_SET_MaxDynamicSharedMemorySize(FUNC, VAL) \
  cudaFuncSetAttribute(FUNC, cudaFuncAttributeMaxDynamicSharedMemorySize, VAL)
#else
#define DevFuncAttribute_SET_MaxDynamicSharedMemorySize(FUNC, VAL) \
  hipFuncSetAttribute(FUNC, hipFuncAttributeMaxDynamicSharedMemorySize, VAL)

  // NOTE(yiakwy) : func alias
  template <typename... Args>
  static __inline__ __host__ __device__
  auto cudaLaunchCooperativeKernel(Args&&... args) -> decltype(hipLaunchCooperativeKernel(std::forward<Args>(args)...)) {
    return hipLaunchCooperativeKernel(std::forward<Args>(args)...);
  }
#endif

#define CEILDIV(x, y) (((x) + (y)-1) / (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))

#define DISPATCH_CASE_INTEGRAL_TYPES(...)              \
  AT_DISPATCH_CASE(at::ScalarType::Byte, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::Char, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::Short, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Int, __VA_ARGS__)   \
  AT_DISPATCH_CASE(at::ScalarType::Long, __VA_ARGS__)

#define DISPATCH_INTEGRAL_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, DISPATCH_CASE_INTEGRAL_TYPES(__VA_ARGS__))

__device__ __forceinline__ int32_t index(int32_t total_col, int32_t row, int32_t col) {
  // don't worry about overflow because num_experts is relatively small
  return row * total_col + col;
}

#define FRAGS_PER_BLOCK  4
#define FRAG_SIZE_M    16
#define FRAG_SIZE_N   16
#define MAX_NUM_EXPERTS 256
#define SHIFT_1_PAD     1

#define USE_CUSUM_LOCAL_CACHE false

namespace cg = cooperative_groups;

template <typename scalar_t>
__global__ void moe_align_block_size_kernel(scalar_t* __restrict__ topk_ids, int32_t* sorted_token_ids,
                                            int32_t* expert_ids, int32_t* total_tokens_post_pad, int32_t num_experts,
                                            int32_t block_size, size_t numel, int32_t* cumsum) {
  __shared__ int32_t shared_counts[32][8];
  // NOTE (yiakwy) : this assumes num_experts <= 256
  __shared__ int32_t local_offsets[256+SHIFT_1_PAD];
  __shared__ int32_t local_offsets_buf[16];

  const int tid = threadIdx.x;
  const int experts_per_warp = 8;

  const size_t tokens_per_thread = CEILDIV(numel, blockDim.x);
  const size_t start_idx = threadIdx.x * tokens_per_thread;

  int *shared_counts_base = &(shared_counts[0][0]);
  if (threadIdx.x < 256) {
    *(shared_counts_base + threadIdx.x) = 0;
  }

  // NOTE (yiakwy) : this warp of threads may access other warp of threads based on the value of expert id fetched
  __syncthreads();

  for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
    int expert_id = topk_ids[i];
    int warp_idx = expert_id / experts_per_warp;
    int expert_offset = expert_id % experts_per_warp;
    atomicAdd(&shared_counts[warp_idx][expert_offset], 1);
  }

  __syncthreads();

#define kElementsPerThr    16

  {

    int active_threads = CEILDIV(num_experts, kElementsPerThr);
    if (tid == 0) {
      local_offsets[0] = 0;
    }
    if (tid < active_threads - 1) { // NOTE(yaikwy) : algo here assumes single block execution

      // NOTE (yiakwy) : loop body, a simple reduction prototype, useful for workload with the number of experts upto 256
      // NOTE (yiakwy) : each thread process 16 expert, then only 2 steps needed

      // NOTE (yiakwy) : step 1, loop body
      for (int i=tid*kElementsPerThr+1; i < (tid + 1)*kElementsPerThr+1; ++i) {
        int warp_idx = (i-1) / experts_per_warp;
        int expert_offset = (i-1) % experts_per_warp;

        int expert_count = shared_counts[warp_idx][expert_offset];

        int last_val = (i-1) % kElementsPerThr == 0 ? 0 : local_offsets[i-1];
        local_offsets[i] = last_val + CEILDIV(expert_count, block_size) * block_size;
      }

      local_offsets_buf[tid] = local_offsets[(tid + 1)*kElementsPerThr];

    }

    // NOTE (yiakwy) : step 1, unroll loop tail
    if (tid == active_threads - 1) {
      #pragma unroll
      for (int i=tid * kElementsPerThr+1; i < num_experts+1; ++i) {
        int warp_idx = (i-1) / experts_per_warp;
        int expert_offset = (i-1) % experts_per_warp;

        int expert_count = shared_counts[warp_idx][expert_offset];

        int last_val = (i-1) % kElementsPerThr == 0 ? 0 : local_offsets[i-1];
        local_offsets[i] = last_val + CEILDIV(expert_count, block_size) * block_size;
      }

      local_offsets_buf[tid] = local_offsets[num_experts];

    }
    __syncthreads();

    // NOTE (yiakwy) : step 2, loop body
    if (tid < active_threads - 1 && tid > 0) {
      int offset = 0;
      for (int j=0; j < tid; ++j) {
        offset += local_offsets_buf[j];
      }

      for (int i=tid*kElementsPerThr+1; i < (tid + 1)*kElementsPerThr+1; ++i) {
        local_offsets[i] += offset;
      }
    }

    // NOTE (yiakwy) : step 2, loop tail
    if (tid == active_threads - 1) {
      int offset = 0;
      for (int j=0; j < tid; ++j) {
        offset += local_offsets_buf[j];
      }
      for (int i=tid*kElementsPerThr+1; i < num_experts+1; ++i) {
        local_offsets[i] += offset;
      }
    }

  } // code block of computing cumsum
  __syncthreads();

#define kElementsPerThr    16
#define kElementsPerAccess 4

  {

    int active_threads = CEILDIV(num_experts+1, kElementsPerThr);
    if (tid < active_threads - 1) {

      // NOTE(yiakwy) : loop body useful for workload with the number of experts upto 256
      for (int i=tid * kElementsPerThr ; i < (tid + 1) * kElementsPerThr; i += kElementsPerAccess) {
        *(int4 *)(cumsum + i) = *(int4 *)(local_offsets + i);
      }
    }

    if (tid == active_threads - 1) {
      // NOTE(yiakwy) : unroll loop tail
      #pragma unroll
      for (int i=tid * kElementsPerThr; i < num_experts+1; i++) {
        *(cumsum + i) = *(local_offsets + i);
      }
    }

    if (tid == active_threads) {
      *total_tokens_post_pad = local_offsets[num_experts];
    }

  } // code block of storing to cumsum
  __syncthreads();

  if (threadIdx.x < num_experts) {
    for (int i = local_offsets[threadIdx.x]; i < local_offsets[threadIdx.x + 1]; i += block_size) {
      expert_ids[i / block_size] = threadIdx.x;
    }
  }
  __syncthreads();

  for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
    int32_t expert_id = topk_ids[i];
    int32_t rank_post_pad = atomicAdd(&local_offsets[expert_id], 1);
    sorted_token_ids[rank_post_pad] = i;
  }
}


#define checkCudaErrors(val) check ( (val), #val, __FILE__, __LINE__ )
template< typename T >
void check(T result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(result), cudaGetErrorString(result), func);
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}


template <typename scalar_t>
__global__ void moe_align_block_size_multiblocks_kernel(scalar_t* __restrict__ topk_ids, int32_t* sorted_token_ids,
                                            int32_t* expert_ids, int32_t* total_tokens_post_pad, int32_t num_experts,
                                            int32_t block_size, size_t numel, int32_t* tokens_cnts, int32_t* cumsum, const int tokens_per_block, const int tokens_per_thread, const int K) {
  // NOTE (yiakwy) : for 16x16 fragment, the maximum of 16 fragments be used in a single block to process MAX_NUM_EXPERTS (256) experts
  __shared__ int32_t smem[ FRAG_SIZE_M * FRAG_SIZE_N* FRAGS_PER_BLOCK ];
  int32_t (*shared_counts)[8] = (int32_t (*)[8])&smem[0];
  // NOTE (yiakwy) : this assumes num_experts <= MAX_NUM_EXPERTS (256)
  __shared__ int32_t local_offsets[ MAX_NUM_EXPERTS + SHIFT_1_PAD];
  // NOTE (yiakwy) : lcoal buf for parallel cumsum
  __shared__ int32_t local_offsets_buf[16];

  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  // NOTE (yiakwy) : we use local warp_id, lane_id for warp aggregation of shared_counts
  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;
  const int experts_per_warp = 8;

  // NOTE (yiakwy) : used to synchronize blocks,
  cg::grid_group grid = cg::this_grid();

  // NOTE (yiakwy) : not all threads participate in
  const size_t start_idx = tokens_per_block * blockIdx.x + tokens_per_thread * threadIdx.x;
  const size_t end_idx = start_idx + tokens_per_thread;

  if (threadIdx.x < FRAG_SIZE_M * FRAG_SIZE_N) {
    for (int i=0; i < FRAG_SIZE_M * FRAG_SIZE_N * FRAGS_PER_BLOCK ; i+=FRAG_SIZE_M * FRAG_SIZE_N) {
      // *(shared_counts_base + threadIdx.x + i) = 0;
      smem[threadIdx.x + i] = 0;
    }
  }
  __syncthreads();

  int *shared_counts_base = &(shared_counts[0][0]);

  // NOTE (yiakwy) : since each block processes less tokens, less possibility for threads acces these localtions ([0][0], [4][0], [8][0], ...) simutaneously
  if (threadIdx.x * tokens_per_thread < tokens_per_block) {
    for (int i = start_idx; i < MIN(numel, end_idx); ++i) {
      int expert_id = topk_ids[i];
      int warp_idx = expert_id / experts_per_warp;
      int expert_offset = expert_id % experts_per_warp;
      atomicAdd(&shared_counts[warp_idx][expert_offset], 1);
    }
  }
  __syncthreads();

#define kElementsPerThr    16

  {

    int active_threads = CEILDIV(num_experts, kElementsPerThr);
    if (threadIdx.x == 0) {
      local_offsets[0] = 0;
    }
    if (threadIdx.x < active_threads - 1) { // NOTE(yaikwy) : algo here assumes single block execution

      // NOTE (yiakwy) : loop body, a simple reduction prototype, useful for workload with the number of experts upto 256
      // NOTE (yiakwy) : each thread process 16 expert, then only 2 steps needed

      // NOTE (yiakwy) : step 1, loop body
      for (int i=threadIdx.x*kElementsPerThr+1; i < (threadIdx.x + 1)*kElementsPerThr+1; ++i) {
        int warp_idx = (i-1) / experts_per_warp;
        int expert_offset = (i-1) % experts_per_warp;

        int expert_count = shared_counts[warp_idx][expert_offset];

        int last_val = (i-1) % kElementsPerThr == 0 ? 0 : local_offsets[i-1];
        // local_offsets[i] = last_val + CEILDIV(expert_count, block_size) * block_size;
        local_offsets[i] = last_val + expert_count;
      }

      local_offsets_buf[threadIdx.x] = local_offsets[(threadIdx.x + 1)*kElementsPerThr];

    }

    // NOTE (yiakwy) : step 1, unroll loop tail
    if (threadIdx.x == active_threads - 1) {
      #pragma unroll
      for (int i=threadIdx.x * kElementsPerThr+1; i < num_experts+1; ++i) {
        int warp_idx = (i-1) / experts_per_warp;
        int expert_offset = (i-1) % experts_per_warp;

        int expert_count = shared_counts[warp_idx][expert_offset];

        int last_val = (i-1) % kElementsPerThr == 0 ? 0 : local_offsets[i-1];
        local_offsets[i] = last_val + expert_count;
      }

      local_offsets_buf[threadIdx.x] = local_offsets[num_experts];

    }
    __syncthreads();

    // NOTE (yiakwy) : step 2, loop body
    if (threadIdx.x < active_threads - 1 && threadIdx.x > 0) {
      int offset = 0;
      for (int j=0; j < threadIdx.x; ++j) {
        offset += local_offsets_buf[j];
      }

      for (int i=threadIdx.x*kElementsPerThr+1; i < (threadIdx.x + 1)*kElementsPerThr+1; ++i) {
        local_offsets[i] += offset;
      }
    }

    // NOTE (yiakwy) : step 2, loop tail
    if (threadIdx.x == active_threads - 1) {
      int offset = 0;
      for (int j=0; j < threadIdx.x; ++j) {
        offset += local_offsets_buf[j];
      }
      for (int i=threadIdx.x*kElementsPerThr+1; i < num_experts+1; ++i) {
        local_offsets[i] += offset;
      }
    }

  } // code block of computing local unaligned cumsum
  __syncthreads();

#ifdef DEBUG
  if (threadIdx.x == 0) {
    printf("[Block#%d/%d] local_offsets[0:num_experts+1] = [%d, %d, %d, ..., %d, %d]\n", blockIdx.x, gridDim.x, local_offsets[0], local_offsets[1], local_offsets[2], local_offsets[num_experts-1], local_offsets[num_experts]);
  }
  __syncthreads();
#endif

  {
    if (tid < num_experts) {
      *(tokens_cnts + tid) = 0;
    }
    if (threadIdx.x < num_experts) {
      *(tokens_cnts + (blockIdx.x + 1) * num_experts + threadIdx.x) = *(local_offsets + threadIdx.x + 1);
      *(local_offsets + threadIdx.x + 1) = 0;
    } else if (threadIdx.x < MAX_NUM_EXPERTS) {
      *(local_offsets + threadIdx.x + 1) = 0;
    }
    __threadfence_system();
    grid.sync();

#define kElementsPerAccess 4
#define kWarpsToLoad       2

    int total_fragments     = CEILDIV(num_experts, FRAG_SIZE_N);
    int fragments_per_block = CEILDIV(total_fragments, gridDim.x);
    int fragments_per_warp  = CEILDIV(fragments_per_block, FRAGS_PER_BLOCK);

#ifdef DEBUG
    if (tid == 0) {
      printf("[tid#0] total fragments : %d, fragments/block : %d, fragments/warp : %d\n", total_fragments, fragments_per_block, fragments_per_warp);
    }
#endif

    int *tokens_cnts_ptr = &(tokens_cnts[0]);

    for (int i=0; i < gridDim.x; i += FRAG_SIZE_M) {
      for ( int j=0; j < fragments_per_warp; j++) {

        if (warp_id * fragments_per_warp < kWarpsToLoad * fragments_per_block) { // NOTE (yiakwy) : kWarpsToLoad warps (CUDA) for loading 16x16 fragment
          const int kNumThrPerRow = WARP_SIZE / FRAG_SIZE_N;
          int sRow = lane_id / kNumThrPerRow ;

          int sWarpColStride = kNumThrPerRow * kElementsPerAccess;
          int sWarpColOff = warp_id * sWarpColStride;
          int sThrColOff = lane_id % kNumThrPerRow * kElementsPerAccess;

          int sCol = sThrColOff + sWarpColOff;

          int gRow = i + sRow;

          int gBlockColOff = blockIdx.x * fragments_per_block * FRAG_SIZE_N;
          int gWarpColOff_0 = (warp_id / kWarpsToLoad * fragments_per_warp + j) * FRAG_SIZE_N;
          int gWarpColOff_1 = warp_id % kWarpsToLoad * sWarpColStride;

          int gCol = gBlockColOff + gWarpColOff_0 + gWarpColOff_1 + sThrColOff;

          if (gRow < num_experts && gCol < num_experts) { // NOTE (yiakwy) : defensive guard
            // NOTE (yiakwy) : useful to coalesce memory transaction when loading a column of data
            int4 *tokens_cnts_4i_ptr = (int4 *)(tokens_cnts_ptr + (gRow+1) * num_experts + gCol);
            int4 *shared_counts_4i_ptr = (int4 *)(shared_counts_base + sRow * FRAGS_PER_BLOCK * FRAG_SIZE_N + sCol);

            *shared_counts_4i_ptr = *tokens_cnts_4i_ptr;
          }
        }
        __syncthreads();

        if (warp_id * fragments_per_warp < kWarpsToLoad * fragments_per_block) {
          if (warp_id % kWarpsToLoad == 0) { // NOTE (yiakwy) : 1x warp (CUDA) for processing 16x16 fragment
            for (int k=0; k < FRAG_SIZE_N; k+=2) { // NOTE (yiakwy) : this simple arangement enables thread 0 accessing addresses limited to bank 0, thread 16 accesses limited to bank 16, etc in CUDA.
              int sRow = lane_id / FRAG_SIZE_N + k;
              int sThrColOff = lane_id % FRAG_SIZE_N;
              int sCol = sThrColOff + (warp_id / kWarpsToLoad) * FRAG_SIZE_N;

              int gBlockColOff = blockIdx.x * fragments_per_block * FRAG_SIZE_N;
              int gWarpColOff_0 = (warp_id / kWarpsToLoad * fragments_per_warp + j) * FRAG_SIZE_N;
              int gCol = gBlockColOff + gWarpColOff_0 + sThrColOff;
              if (gCol < num_experts) { // NOTE (yiakwy) : defensive guard
                atomicAdd(tokens_cnts_ptr + gCol, *(shared_counts_base + sRow * FRAGS_PER_BLOCK * FRAG_SIZE_N + sCol));
              }
            }
          }
        }
        __syncthreads();

      } // end of j

    } // end of i

    __threadfence_system();
    grid.sync();

    // NOTE (yiakwy) : sync unaligned offsets to blocks#0
    // TODO (yiakwy) : distribute work to other threads
    if (tid < num_experts) {
      *(local_offsets + tid + 1) = *(tokens_cnts + tid);
    }
    __syncthreads();

#ifdef DEBUG
    if (tid == 0) {
      printf("[Block#%d] unaligned global offsets[0:num_experts+1] = [%d, %d, %d, ..., %d, %d]\n", blockIdx.x, local_offsets[0], local_offsets[1], local_offsets[2], local_offsets[num_experts-1], local_offsets[num_experts]);
    }
    __syncthreads();
#endif

  } // code block of computing global cumsum
  __syncthreads();

  // NOTE (yiakwy) : convert unaligned cumsum to aligned cumsum
  // TODO (yiakwy) : distribute work to other threads
  if (tid == 0) {
    for (int i=num_experts; i > 0; i--) {
      local_offsets[i] = local_offsets[i] - local_offsets[i-1];
    }
    for (int i=1; i < num_experts+1; i++) {
      local_offsets[i] = local_offsets[i-1] + CEILDIV(local_offsets[i], block_size) * block_size;
    }
  }
  __syncthreads();

#ifdef DEBUG
  if (tid == 0) {
    printf("[Block#%d] aligned global offsets[0:num_experts+1] = [%d, %d, %d, ..., %d, %d]\n", blockIdx.x, local_offsets[0], local_offsets[1], local_offsets[2], local_offsets[num_experts-1], local_offsets[num_experts]);
  }
  __syncthreads();
#endif

#define kElementsPerThr    16
#define kElementsPerAccess 4

  {

    int active_threads = CEILDIV(num_experts+1, kElementsPerThr);
    if (tid < active_threads - 1) {

      // NOTE(yiakwy) : loop body useful for workload with the number of experts upto 256
      for (int i=tid * kElementsPerThr ; i < (tid + 1) * kElementsPerThr; i += kElementsPerAccess) {
        *(int4 *)(cumsum + i) = *(int4 *)(local_offsets + i);
#ifdef DEBUG
        *(int4 *)(tokens_cnts + i) = *(int4 *)(local_offsets + i);
#endif
      }

      // printf("[tid#%d] cumsum[%d] (%d) = local_offsets[%d] (%d %d %d %d ...)\n", tid, tid * kElementsPerThr, cumsum[tid * kElementsPerThr], tid * kElementsPerThr, local_offsets[tid * kElementsPerThr], local_offsets[tid * kElementsPerThr+1], local_offsets[tid * kElementsPerThr+2], local_offsets[tid * kElementsPerThr+3]);
    }

    if (tid == active_threads - 1) {
      // NOTE(yiakwy) : unroll loop tail
      #pragma unroll
      for (int i=tid * kElementsPerThr; i < num_experts+1; i++) {
        *(cumsum + i) = *(local_offsets + i);
#ifdef DEBUG
        *(tokens_cnts + i) = *(local_offsets + i);
#endif
      }
    }

    if (tid == active_threads) {
      *total_tokens_post_pad = local_offsets[num_experts];
    }

  } // code block of storing to cumsum
  __threadfence_system();
  grid.sync();

  if (USE_CUSUM_LOCAL_CACHE) {
  // NOTE (yiakwy) : sync cumsum to each block
    if (blockIdx.x > 0) {
      int active_threads = CEILDIV(num_experts+1, kElementsPerThr);

      if (threadIdx.x < active_threads - 1) {
        for (int i=threadIdx.x * kElementsPerThr ; i < (threadIdx.x + 1) * kElementsPerThr; i += kElementsPerAccess) {
          *(int4 *)(local_offsets + i) = *(int4 *)(cumsum + i);
        }
      }

      if (threadIdx.x == active_threads - 1) {
        #pragma unroll
        for (int i=threadIdx.x * kElementsPerThr; i < num_experts+1; i++) {
          *(local_offsets + i) = *(cumsum + i);
        }
      }
    }
  }
  __threadfence_system();
  grid.sync();

  if (tid < num_experts) {
    for (int i = local_offsets[tid]; i < local_offsets[tid + 1]; i += block_size) {
      expert_ids[i / block_size] = tid;
    }
  }
  __syncthreads();

  if (threadIdx.x * tokens_per_thread < tokens_per_block) {
    for (int i = start_idx; i < MIN(numel, end_idx); ++i) {
      if (USE_CUSUM_LOCAL_CACHE) {
        int rank_post_pad, old=i, tok=i, expert_id = topk_ids[i];
        do {
          rank_post_pad = atomicAdd(&local_offsets[expert_id], 1);
          old = atomicCAS(&sorted_token_ids[rank_post_pad], numel, tok);
        } while (old != numel && old != tok);
      } else {
        int32_t expert_id = topk_ids[i];
#ifdef DEBUG
        int32_t rank_post_pad = atomicAdd(&tokens_cnts[expert_id], 1);
#else
        int32_t rank_post_pad = atomicAdd(&cumsum[expert_id], 1);
#endif
        sorted_token_ids[rank_post_pad] = i;
      }
    }
  }
}


void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts, int64_t block_size,
                          torch::Tensor sorted_token_ids, torch::Tensor experts_ids, torch::Tensor num_tokens_post_pad,
                          torch::Tensor token_cnts_buffer, torch::Tensor cumsum_buffer) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  DISPATCH_INTEGRAL_TYPES(topk_ids.scalar_type(), "moe_align_block_size_kernel", [&] {
    if (false/*topk_ids.sizes()[0] < 16384 / 2*/) {
      auto kernel = moe_align_block_size_kernel<scalar_t>;
      // NOTE(yiakwy) : this assumes a single block execution, will be slow if too many tokens (>1024) feeded in
      kernel<<<1, 1024, 0, stream>>>(topk_ids.data_ptr<scalar_t>(), sorted_token_ids.data_ptr<int32_t>(),
                                    experts_ids.data_ptr<int32_t>(), num_tokens_post_pad.data_ptr<int32_t>(),
                                    num_experts, block_size, topk_ids.numel(), cumsum_buffer.data_ptr<int32_t>());
    } else {
      auto kernel = moe_align_block_size_multiblocks_kernel<scalar_t>;
// NOTE (yiakwy) : reduce registers consumed
#define BLOCK_SIZE 256
      auto BLOCKS = MIN( CEILDIV(topk_ids.sizes()[0], BLOCK_SIZE), num_experts );

      int32_t tokens_per_block = CEILDIV(topk_ids.sizes()[0], BLOCKS) * topk_ids.sizes()[1];
      int32_t tokens_per_thread = CEILDIV(tokens_per_block, BLOCK_SIZE);

      // NOTE (yiakwy) : remove const decorator for kernel args
      scalar_t* topk_ids_ptr = topk_ids.data_ptr<scalar_t>();
      int32_t* sorted_token_ids_ptr = sorted_token_ids.data_ptr<int32_t>();
      int32_t* experts_ids_ptr = experts_ids.data_ptr<int32_t>();
      int32_t* num_tokens_post_pad_ptr = num_tokens_post_pad.data_ptr<int32_t>();
      size_t num_tokens = topk_ids.numel();
      int32_t* token_cnts_buffer_ptr = token_cnts_buffer.data_ptr<int32_t>();
      int32_t* cumsum_buffer_ptr = cumsum_buffer.data_ptr<int32_t>();
      int K = topk_ids.sizes()[1];

      void *kernelArgs[] = { &topk_ids_ptr, &sorted_token_ids_ptr,
        &experts_ids_ptr, &num_tokens_post_pad_ptr,
        &num_experts, &block_size, &num_tokens, &token_cnts_buffer_ptr, &cumsum_buffer_ptr, &tokens_per_block, &tokens_per_thread, &K
      };
      cudaLaunchCooperativeKernel((void*)kernel, BLOCKS, BLOCK_SIZE, kernelArgs);
#ifdef DEBUG
      checkCudaErrors(cudaDeviceSynchronize());
#endif
    }
  });
}
