#include "m1_intrinsics.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

bool log_case(std::string_view name, std::size_t index, float actual, std::uint32_t expected, bool verbose) {
  const auto actual_bits = std::bit_cast<std::uint32_t>(actual);
  const bool passed = actual_bits == expected;
  if (verbose || !passed) {
    std::cout << "  " << name << "[" << index << "] " << std::hex << std::showbase;
    std::cout << "actual=" << actual_bits << " expected=" << expected << std::dec << std::noshowbase;
    std::cout << (passed ? " OK" : " FAIL") << '\n';
  }
  return passed;
}

bool run_e2m1_checks(bool verbose) {
  bool success = true;
  const auto& expected = sm120::unit::kE2M1ExpectedBits;
  std::cout << "Running " << expected.size() << " E2M1 points" << (verbose ? " (verbose)" : "") << "\n";
  for (std::size_t idx = 0; idx < expected.size(); ++idx) {
    const float actual = sm120::unit::decode_e2m1(static_cast<std::uint8_t>(idx));
    success &= log_case("E2M1", idx, actual, expected[idx], verbose);
  }
  return success;
}

bool run_ue4m3_checks(bool verbose) {
  bool success = true;
  const auto& expected = sm120::unit::kUE4M3ExpectedBits;
  std::cout << "Running " << expected.size() << " UE4M3 checks" << (verbose ? " (verbose)" : "") << "\n";
  for (std::size_t idx = 0; idx < expected.size(); ++idx) {
    const float actual = sm120::unit::decode_ue4m3(static_cast<std::uint8_t>(idx));
    success &= log_case("UE4M3", idx, actual, expected[idx], verbose);
  }
  return success;
}

struct BFloat16 {
  std::uint16_t bits = 0;

  constexpr BFloat16() noexcept = default;
  explicit constexpr BFloat16(float value) noexcept : bits(to_bfloat_bits(value)) {}

  constexpr float to_float() const noexcept {
    const std::uint32_t widened = static_cast<std::uint32_t>(bits) << 16;
    return std::bit_cast<float>(widened);
  }

  constexpr operator float() const noexcept { return to_float(); }

 private:
  static constexpr std::uint16_t to_bfloat_bits(float value) noexcept {
    const std::uint32_t raw = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t truncated = raw >> 16;
    const std::uint32_t sticky = raw & 0xFFFFu;
    std::uint32_t result = truncated;

    const std::uint32_t round_bias = 0x8000u;
    if (sticky > round_bias || (sticky == round_bias && (truncated & 1u) != 0u)) {
      result += 1u;
    }

    return static_cast<std::uint16_t>(result);
  }
};

constexpr std::array<int, 3> kHiddenSizes = {512, 1024, 2048};
constexpr std::array<const char*, 3> kM2TestNames = {
    "m2_b_bf16_reference_d512",
    "m2_b_bf16_reference_d1024",
    "m2_b_bf16_reference_d2048",
};

bool run_bf16_reference_test(int hidden_size, bool verbose) {
  std::vector<float> input(hidden_size);
  for (int i = 0; i < hidden_size; ++i) {
    input[i] = std::sin(0.0174532924f * static_cast<float>(i + 1)) * 0.78f;
  }

  constexpr float kScale = 1.375f;
  constexpr float kBias = 0.125f;

  std::vector<BFloat16> encoded(hidden_size);
  std::vector<BFloat16> processed(hidden_size);

  for (int i = 0; i < hidden_size; ++i) {
    encoded[i] = BFloat16(input[i]);
    const float stage = static_cast<float>(encoded[i]) * kScale + kBias;
    processed[i] = BFloat16(stage);
  }

  float max_error = 0.0f;
  float sum_error = 0.0f;
  for (int i = 0; i < hidden_size; ++i) {
    const float expected = input[i] * kScale + kBias;
    const float actual = static_cast<float>(processed[i]);
    const float diff = std::fabs(expected - actual);
    if (diff > max_error) {
      max_error = diff;
    }
    sum_error += diff;
  }

  const float mean_error = sum_error / static_cast<float>(hidden_size);
  const float tolerance = 0.03f;

  if (verbose) {
    std::cout << "    M2-B BF16 reference d=" << hidden_size << " max_err=" << max_error
              << " mean_err=" << mean_error << "\n";
  }

  return max_error <= tolerance;
}

bool run_m2_bf16_checks(bool verbose) {
  bool all_passed = true;
  for (std::size_t idx = 0; idx < kHiddenSizes.size(); ++idx) {
    const bool passed = run_bf16_reference_test(kHiddenSizes[idx], verbose);
    all_passed &= passed;
    std::cout << "  [" << (passed ? "PASS" : "FAIL") << "] " << kM2TestNames[idx] << "\n";
  }
  return all_passed;
}

}  // namespace

int main(int argc, char** argv) {
  bool verbose = false;
  bool list_only = false;
  std::string mode = "default";

  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    if (arg == "--verbose" || arg == "-v") {
      verbose = true;
    } else if (arg == "--list") {
      list_only = true;
    } else if (arg == "--mode" && i + 1 < argc) {
      mode = argv[++i];
    }
  }

  if (list_only) {
    std::cout << "SM_120 unit harness: available tests -> [m1_intrinsics_e2m1, m1_intrinsics_ue4m3, "
                 "m2_b_bf16_reference_d512, m2_b_bf16_reference_d1024, m2_b_bf16_reference_d2048]\n";
    return 0;
  }

  if (mode != "default" && mode != "m1" && mode != "m2" && mode != "e2m1" && mode != "ue4m3") {
    std::cerr << "Unknown mode '" << mode << "'. Valid modes: default, m1, m2, e2m1, ue4m3\n";
    return 1;
  }

  std::cout << "SM_120 unit harness running in '" << mode << "' mode";
  std::cout << (verbose ? " (verbose)" : "") << "\n";

  bool overall = true;
  if (mode == "default" || mode == "m1" || mode == "e2m1") {
    overall &= run_e2m1_checks(verbose);
  }
  if (mode == "default" || mode == "m1" || mode == "ue4m3") {
    overall &= run_ue4m3_checks(verbose);
  }
  if (mode == "default" || mode == "m2") {
    overall &= run_m2_bf16_checks(verbose);
  }

  std::cout << "Suite " << (overall ? "PASSED" : "FAILED")
            << " (16 E2M1 points, 256 UE4M3 checks, 3 BF16 reference sizes)\n";
  return overall ? 0 : 1;
}
