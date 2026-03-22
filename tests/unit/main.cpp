#include "m1_intrinsics.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>

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

}  // namespace

int main(int argc, char** argv) {
  bool verbose = false;
  bool list_only = false;
  std::string mode = "default";

  for (int i = 1; i < argc; ++i) {
    std::string_view arg(argv[i]);
    if (arg == "--verbose" || arg == "-v") {
      verbose = true;
    } else if (arg == "--list") {
      list_only = true;
    } else if (arg == "--mode" && i + 1 < argc) {
      mode = argv[++i];
    }
  }

  const auto list_tests = [] {
    std::cout << "SM_120 unit harness: available tests -> [m1_intrinsics_e2m1, m1_intrinsics_ue4m3]\n";
  };

  if (list_only) {
    list_tests();
    return 0;
  }

  if (mode != "default" && mode != "e2m1" && mode != "ue4m3") {
    std::cerr << "Unknown mode '" << mode << "'. Valid modes: default, e2m1, ue4m3\n";
    return 1;
  }

  std::cout << "SM_120 M1 intrinsic checks running in '" << mode << "' mode";
  std::cout << (verbose ? " (verbose)" : "") << "\n";

  bool overall = true;
  if (mode == "default" || mode == "e2m1") {
    overall &= run_e2m1_checks(verbose);
  }
  if (mode == "default" || mode == "ue4m3") {
    overall &= run_ue4m3_checks(verbose);
  }

  std::cout << "M1 intrinsic suite " << (overall ? "PASSED" : "FAILED") << " (16 E2M1 points, 256 UE4M3 checks)\n";
  return overall ? 0 : 1;
}
