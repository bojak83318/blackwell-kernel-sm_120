extern "C" __global__ void tcgen05_probe() {
  asm volatile(
      "tcgen05.mma.sync.aligned.m16n8k16.row_col_n1.nvf4.nvf4.f32.row_f32 "
      "[%0], [%1], [%2], {%3, %4};"
      :
      : "r"(0), "l"(0), "l"(0), "r"(0.0f), "r"(1.0f)
      : "memory");
}
