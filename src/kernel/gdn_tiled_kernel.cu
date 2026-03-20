#include <cstdint>

extern "C" __global__ void gdn_tiled_moe_kernel(
    const std::uint8_t* a_sparse,
    const std::uint8_t* b_dense,
    float* c_accum,
    const std::uint8_t* sparse_meta,
    int m,
    int n,
    int k) {
  (void)a_sparse;
  (void)b_dense;
  (void)c_accum;
  (void)sparse_meta;
  (void)m;
  (void)n;
  (void)k;

  const std::uint64_t ptr = 0;
  const float acc0 = 0.0f;
  const float acc1 = 1.0f;
  asm volatile(
      "tcgen05.mma.sync.aligned.m16n8k16.row_col_n1.nvf4.nvf4.f32.row_f32 "
      "[%0], [%1], [%2], {%3, %4};"
      :
      : "l"(ptr), "l"(ptr), "l"(ptr), "f"(acc0), "f"(acc1)
      : "memory");
}
