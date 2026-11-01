// legacy/quantization, with vanilla-FA3 per-head (per-(b,h) scale) fp8 quantizer as harnessing reference.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace {

constexpr int kHead = 128;
constexpr int kHalf = 64;   // one 64-wide H64 block per 8-lane half-warp
constexpr int kThreads = 256;
constexpr int kLanesPerTok = 16;
constexpr unsigned kFullMask = 0xffffffffu;

// H64 Walsh-Hadamard (Sylvester order) on one 64-wide half, normalized by
// 1/sqrt(64) = 1/8.
__device__ __forceinline__ void fwht64(float v[8], int lane) {
#pragma unroll
  for (int step = 1; step < 8; step <<= 1)
#pragma unroll
    for (int r = 0; r < 8; ++r)
      if ((r & step) == 0) {
        const float a = v[r], b = v[r + step];
        v[r] = a + b;
        v[r + step] = a - b;
      }
#pragma unroll
  for (int step = 8; step < kHalf; step <<= 1) {
    const int src = step >> 3;
    // (lane & src) == 0 ? v + o : o - v  ==  fma(+/-1, v, o): one FFMA per
    // value instead of add/sub/select, and bit-exact with the branch form.
    const float sgn = (lane & src) ? -1.0f : 1.0f;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
      const float o = __shfl_xor_sync(0xffffffffu, v[r], src, 8);
      v[r] = fmaf(sgn, v[r], o);
    }
  }
  constexpr float kInv = 1.0f / 8.0f;
#pragma unroll
  for (int r = 0; r < 8; ++r) v[r] *= kInv;
}

template <bool IN_BF16>
__device__ __forceinline__ void load8(const void* xp, long tok, int lane,
                                      float v[8]) {
  const uint4 raw = __ldcs(reinterpret_cast<const uint4*>(
      reinterpret_cast<const __half*>(xp) + tok * kHead) + lane);
  const unsigned short* hs = reinterpret_cast<const unsigned short*>(&raw);
#pragma unroll
  for (int r = 0; r < 8; ++r)
    v[r] = IN_BF16 ? __bfloat162float(__ushort_as_bfloat16(hs[r]))
                   : __half2float(__ushort_as_half(hs[r]));
}

__device__ __forceinline__ unsigned short cvt_e4m3x2(float hi, float lo) {
  unsigned short d;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
      : "=h"(d) : "f"(hi), "f"(lo));
#else
  const __nv_fp8_e4m3 qh(hi), ql(lo);
  d = (unsigned short)(*reinterpret_cast<const unsigned char*>(&qh) << 8 |
                       *reinterpret_cast<const unsigned char*>(&ql));
#endif
  return d;
}

__device__ __forceinline__ unsigned long long pack_e4m3(const float v[8],
                                                        float isc) {
  const unsigned short p0 = cvt_e4m3x2(v[1] * isc, v[0] * isc);
  const unsigned short p1 = cvt_e4m3x2(v[3] * isc, v[2] * isc);
  const unsigned short p2 = cvt_e4m3x2(v[5] * isc, v[4] * isc);
  const unsigned short p3 = cvt_e4m3x2(v[7] * isc, v[6] * isc);
  const unsigned lo = (unsigned)p0 | ((unsigned)p1 << 16);
  const unsigned hi = (unsigned)p2 | ((unsigned)p3 << 16);
  return (unsigned long long)lo | ((unsigned long long)hi << 32);
}

__device__ __forceinline__ void load_off8(const float* __restrict__ p,
                                          float o[8]) {
  const float4 a = *reinterpret_cast<const float4*>(p);
  const float4 b = *reinterpret_cast<const float4*>(p + 4);
  o[0] = a.x; o[1] = a.y; o[2] = a.z; o[3] = a.w;
  o[4] = b.x; o[5] = b.y; o[6] = b.z; o[7] = b.w;
}

// Single-pass quantizer (per-16 or per-token scale). 
template <bool IN_BF16, bool HAS_OFFSET, bool HAS_QOFF, bool ROTATE>
__global__ __launch_bounds__(kThreads) void center_tok_kernel(
    const void* __restrict__ xp, const float* __restrict__ offset,
    const float* __restrict__ q_offset,
    float* __restrict__ amax, float* __restrict__ corr, int N, int S, int H) {
  const int tok = (blockIdx.x * (blockDim.x >> 4)) + (threadIdx.x >> 4);
  const int lane = threadIdx.x & 15;
  const bool valid = tok < N;
  const int t = valid ? tok : 0;

  float v[8];
  load8<IN_BF16>(xp, t, lane, v);
  if constexpr (ROTATE) fwht64(v, lane);

  const int h = HAS_OFFSET ? (t % H) : 0;
  if (HAS_OFFSET) {
    float o[8];
    load_off8(offset + (long)h * kHead + lane * 8, o);
#pragma unroll
    for (int r = 0; r < 8; ++r) v[r] -= o[r];
  }

  if (HAS_QOFF) {
    float qo[8];
    load_off8(q_offset + (long)h * kHead + lane * 8, qo);
    float c = 0.0f;
#pragma unroll
    for (int r = 0; r < 8; ++r) c = fmaf(v[r], qo[r], c);
#pragma unroll
    for (int o = 8; o > 0; o >>= 1)
      c += __shfl_xor_sync(0xffffffffu, c, o, kLanesPerTok);
    if (lane == 0 && valid) corr[t] = c;
  }

  float m = 0.0f;
#pragma unroll
  for (int r = 0; r < 8; ++r) m = fmaxf(m, fabsf(v[r]));
#pragma unroll
  for (int o = 8; o > 0; o >>= 1)
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o, kLanesPerTok));
  if (lane == 0 && valid) amax[t] = m;

}

template <bool IN_BF16, bool HAS_OFF, bool ROTATE>
__global__ __launch_bounds__(kThreads) void quant_direct_kernel(
    const void* __restrict__ xp, const float* __restrict__ offset,
    const float* __restrict__ inv, unsigned char* __restrict__ q8,
    int N, int S, int H) {
  const int tok = (blockIdx.x * (blockDim.x >> 4)) + (threadIdx.x >> 4);
  const int lane = threadIdx.x & 15;
  if (tok >= N) return;
  float v[8];
  load8<IN_BF16>(xp, tok, lane, v);
  if constexpr (ROTATE) fwht64(v, lane);
  if constexpr (HAS_OFF) {
    const int h = tok % H;
    float o[8];
    load_off8(offset + (long)h * kHead + lane * 8, o);
#pragma unroll
    for (int r = 0; r < 8; ++r) v[r] -= o[r];
  }
  const int b = tok / (S * H);
  const float isc = inv[(long)b * H + (tok % H)];
  reinterpret_cast<unsigned long long*>(q8)[(long)tok * kLanesPerTok + lane] =
      pack_e4m3(v, isc);
}

// Per-(b,h) amax to (scale, inv).
__global__ __launch_bounds__(kThreads) void perhead_scale_kernel(
    const float* __restrict__ amax, float* __restrict__ scale,
    float* __restrict__ inv, int S, int H) {
  const int bh = blockIdx.x;
  const int b = bh / H, h = bh % H;
  const long base = (long)b * S * H + h;
  float m = 0.0f;
  for (int i = threadIdx.x; i < S; i += blockDim.x)
    m = fmaxf(m, amax[base + (long)i * H]);
  for (int o = 16; o > 0; o >>= 1)
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  __shared__ float sm[32];
  if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = m;
  __syncthreads();
  if (threadIdx.x < 32) {
    const int nw = blockDim.x >> 5;
    float v = (threadIdx.x < nw) ? sm[threadIdx.x] : 0.0f;
    for (int o = 16; o > 0; o >>= 1)
      v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    if (threadIdx.x == 0) {
      const float mx = fmaxf(v, 1e-12f);
      scale[bh] = mx / 448.0f;
      inv[bh] = 448.0f / mx;
    }
  }
}

// Per-(b,h) scale from an already-reduced amax[B*H] (bulk path).
__global__ __launch_bounds__(kThreads) void head_scale_kernel(
    const float* __restrict__ amax, float* __restrict__ scale,
    float* __restrict__ inv, int BH) {
  const int bh = blockIdx.x * blockDim.x + threadIdx.x;
  if (bh >= BH) return;
  const float mx = fmaxf(amax[bh], 1e-12f);
  scale[bh] = mx / 448.0f;
  inv[bh] = 448.0f / mx;
}

template <bool IN_BF16>
__global__ __launch_bounds__(kThreads) void preprocess_v_fp8_kernel(
    const void* __restrict__ v, uint8_t* __restrict__ fp8_out,
    float* __restrict__ scale_out, int B, int S, int H, float bound) {
  const int N = B * S * H;
  const int tok = (blockIdx.x * (blockDim.x >> 4)) + (threadIdx.x >> 4);
  const int lane = threadIdx.x & 15;
  const bool valid = tok < N;
  const int t = valid ? tok : 0;

  float val[8];
  load8<IN_BF16>(v, t, lane, val);

  float m = 0.0f;
#pragma unroll
  for (int r = 0; r < 8; ++r) m = fmaxf(m, fabsf(val[r]));
#pragma unroll
  for (int o = 8; o > 0; o >>= 1)
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o, kLanesPerTok));
  const float sc = fmaxf(m / bound, 1e-12f);
  const float isc = 1.0f / sc;
  if (lane == 0 && valid) scale_out[t] = sc;

  if (valid)
    __stcs(reinterpret_cast<unsigned long long*>(fp8_out) +
               (long)t * kLanesPerTok + lane,
           pack_e4m3(val, isc));
}

inline void check_x(const torch::Tensor& x) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "x must be contiguous CUDA");
  TORCH_CHECK((x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16) &&
                  x.dim() == 4 && x.size(3) == kHead,
              "x must be fp16/bf16 [B,S,H,128]");
}


template <bool IN_BF16, bool HAS_OFF, bool HAS_Q, bool ROTATE>
inline void launch_center(const torch::Tensor& x, const float* off,
                          const float* qoff, torch::Tensor& amax,
                          torch::Tensor& corr, int N, int S, int H,
                          cudaStream_t stream) {
  const int grid = (N + kThreads / kLanesPerTok - 1) / (kThreads / kLanesPerTok);
  center_tok_kernel<IN_BF16, HAS_OFF, HAS_Q, ROTATE><<<grid, kThreads, 0, stream>>>(
      x.data_ptr(), off, qoff, amax.data_ptr<float>(),
      (corr.numel() ? corr.data_ptr<float>() : nullptr), N, S, H);
}

template <bool IN_BF16, bool HAS_OFF>
inline void launch_quant_direct(bool rotate, const torch::Tensor& x,
                                const float* off, const float* inv,
                                torch::Tensor& q8, int N, int S, int H,
                                cudaStream_t stream) {
  const int grid = (N + kThreads / kLanesPerTok - 1) / (kThreads / kLanesPerTok);
  if (rotate)
    quant_direct_kernel<IN_BF16, HAS_OFF, true><<<grid, kThreads, 0, stream>>>(
        x.data_ptr(), off, inv, reinterpret_cast<unsigned char*>(q8.data_ptr()),
        N, S, H);
  else
    quant_direct_kernel<IN_BF16, HAS_OFF, false><<<grid, kThreads, 0, stream>>>(
        x.data_ptr(), off, inv, reinterpret_cast<unsigned char*>(q8.data_ptr()),
        N, S, H);
}

template <bool IN_BF16, bool HAS_OFF>
inline void launch_center_q(bool has_q, bool rotate, const torch::Tensor& x,
                            const float* off, const float* qoff,
                            torch::Tensor& amax, torch::Tensor& corr, int N,
                            int S, int H, cudaStream_t stream) {
  if (rotate) {
    if (has_q)
      launch_center<IN_BF16, HAS_OFF, true, true>(x, off, qoff, amax, corr, N, S, H, stream);
    else
      launch_center<IN_BF16, HAS_OFF, false, true>(x, off, qoff, amax, corr, N, S, H, stream);
  } else {
    if (has_q)
      launch_center<IN_BF16, HAS_OFF, true, false>(x, off, qoff, amax, corr, N, S, H, stream);
    else
      launch_center<IN_BF16, HAS_OFF, false, false>(x, off, qoff, amax, corr, N, S, H, stream);
  }
}

inline void launch_center_dispatch(bool has_off, bool has_q, bool rotate,
                                   const torch::Tensor& x, const float* off,
                                   const float* qoff, torch::Tensor& amax,
                                   torch::Tensor& corr, int N, int S, int H,
                                   cudaStream_t stream) {
  const bool bf16 = x.scalar_type() == at::kBFloat16;
  if (bf16) {
    if (has_off) launch_center_q<true, true>(has_q, rotate, x, off, qoff, amax, corr, N, S, H, stream);
    else launch_center_q<true, false>(has_q, rotate, x, off, qoff, amax, corr, N, S, H, stream);
  } else {
    if (has_off) launch_center_q<false, true>(has_q, rotate, x, off, qoff, amax, corr, N, S, H, stream);
    else launch_center_q<false, false>(has_q, rotate, x, off, qoff, amax, corr, N, S, H, stream);
  }
}

}  // namespace

// Legacy vanilla-FA3 per-head quantizer (2-pass): consumed only by
// fp8_prefill_h64_fa3.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> quant_head_impl(
    const torch::Tensor& x, const torch::Tensor& offset,
    const torch::Tensor& q_offset, bool want_corr, bool rotate) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() &&
                  (x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16) &&
                  x.dim() == 4 && x.size(3) == kHead,
              "x must be contiguous fp16/bf16 [B,S,H,128]");
  const int B = (int)x.size(0), S = (int)x.size(1), H = (int)x.size(2);
  const bool has_off = offset.numel() > 0;
  const bool has_q = q_offset.numel() > 0;
  TORCH_CHECK(!has_off || (offset.is_cuda() && offset.is_contiguous() &&
                           offset.scalar_type() == at::kFloat && offset.size(-1) == kHead),
              "offset must be contiguous fp32 [H,128] or [B,H,128]");
  TORCH_CHECK(!has_q || (q_offset.is_cuda() && q_offset.is_contiguous() &&
                         q_offset.scalar_type() == at::kFloat && q_offset.size(-1) == kHead),
              "q_offset must be contiguous fp32");
  const int N = B * S * H;
  auto q8 = torch::empty_like(x, x.options().dtype(at::kByte));
  auto scale = torch::empty({B, H}, x.options().dtype(at::kFloat));
  torch::Tensor corr;
  if (want_corr && has_q) corr = torch::empty({B, S, H}, x.options().dtype(at::kFloat));
  else corr = torch::empty({0}, x.options().dtype(at::kFloat));
  const float* off = has_off ? offset.data_ptr<float>() : nullptr;
  const float* qoff = has_q ? q_offset.data_ptr<float>() : nullptr;
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  // Direct 2-pass (no fp16 tmp round-trip): pass 1 rotate+center+amax,
  // per-(b,h) reduce, pass 2 re-read + re-rotate + quantize.
  auto amax = torch::empty({N}, x.options().dtype(at::kFloat));
  launch_center_dispatch(has_off, has_q, rotate, x, off, qoff, amax,
                         corr, N, S, H, stream);

  auto inv = torch::empty({B * H}, x.options().dtype(at::kFloat));
  perhead_scale_kernel<<<B * H, kThreads, 0, stream>>>(
      amax.data_ptr<float>(), scale.data_ptr<float>(), inv.data_ptr<float>(), S, H);

  const bool bf16 = x.scalar_type() == at::kBFloat16;
  if (bf16) {
    if (has_off)
      launch_quant_direct<true, true>(rotate, x, off, inv.data_ptr<float>(), q8, N, S, H, stream);
    else
      launch_quant_direct<true, false>(rotate, x, off, inv.data_ptr<float>(), q8, N, S, H, stream);
  } else {
    if (has_off)
      launch_quant_direct<false, true>(rotate, x, off, inv.data_ptr<float>(), q8, N, S, H, stream);
    else
      launch_quant_direct<false, false>(rotate, x, off, inv.data_ptr<float>(), q8, N, S, H, stream);
  }
  return {q8, scale, corr};
}

std::tuple<torch::Tensor, torch::Tensor> preprocess_q_head_fp8_cuda(
    const torch::Tensor& q, const torch::Tensor& q_offset) {
  auto [q8, scale, corr] = quant_head_impl(q, q_offset, torch::empty({0}, q.options()), false, true);
  (void)corr;
  return {q8, scale};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> preprocess_k_head_fp8_cuda(
    const torch::Tensor& k, const torch::Tensor& q_offset,
    const torch::Tensor& k_offset) {
  return quant_head_impl(k, k_offset, q_offset, true, true);
}

std::tuple<torch::Tensor, torch::Tensor> preprocess_v_head_fp8_cuda(
    const torch::Tensor& v) {
  auto [q8, scale, corr] = quant_head_impl(v, torch::empty({0}, v.options()),
                                           torch::empty({0}, v.options()),
                                           false, true);
  (void)corr;
  return {q8, scale};
}

std::tuple<torch::Tensor, torch::Tensor> preprocess_v_head_fp8_cuda_norot(
    const torch::Tensor& v) {
  auto [q8, scale, corr] = quant_head_impl(v, torch::empty({0}, v.options()),
                                           torch::empty({0}, v.options()),
                                           false, false);
  (void)corr;
  return {q8, scale};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "legacy: FA3 per-head (per-(b,h)) fp8 quantizer baseline";
  m.def("preprocess_q_head_fp8_cuda", &preprocess_q_head_fp8_cuda);
  m.def("preprocess_k_head_fp8_cuda", &preprocess_k_head_fp8_cuda);
  m.def("preprocess_v_head_fp8_cuda", &preprocess_v_head_fp8_cuda);
  m.def("preprocess_v_head_fp8_cuda_norot", &preprocess_v_head_fp8_cuda_norot);
}
