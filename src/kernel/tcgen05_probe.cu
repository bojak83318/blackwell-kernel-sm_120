#include <cstdint>

extern "C" __global__ void tcgen05_probe() {
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
