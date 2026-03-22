#pragma once

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>

namespace sm120::unit {

[[nodiscard]] constexpr std::uint32_t canonical_zero_bits(bool sign) noexcept {
  return sign ? 0x80000000u : 0x00000000u;
}

[[nodiscard]] constexpr std::uint32_t canonical_inf_bits(bool sign) noexcept {
  return sign ? 0xff800000u : 0x7f800000u;
}

[[nodiscard]] constexpr std::uint32_t canonical_nan_bits(bool sign) noexcept {
  return sign ? 0xffc00000u : 0x7fc00000u;
}

constexpr std::array<std::uint32_t, 16> kE2M1ExpectedBits = {
    0x00000000u, 0x3f000000u, 0x3f800000u, 0x3fc00000u, 0x40000000u, 0x40400000u, 0x7f800000u, 0x7fc00000u,
    0x80000000u, 0xbf000000u, 0xbf800000u, 0xbfc00000u, 0xc0000000u, 0xc0400000u, 0xff800000u, 0xffc00000u};

constexpr std::array<std::uint32_t, 256> kUE4M3ExpectedBits = {
    0x00000000u, 0x3b000000u, 0x3b800000u, 0x3bc00000u, 0x3c000000u, 0x3c200000u, 0x3c400000u, 0x3c600000u,
    0x3c800000u, 0x3c900000u, 0x3ca00000u, 0x3cb00000u, 0x3cc00000u, 0x3cd00000u, 0x3ce00000u, 0x3cf00000u,
    0x3d000000u, 0x3d100000u, 0x3d200000u, 0x3d300000u, 0x3d400000u, 0x3d500000u, 0x3d600000u, 0x3d700000u,
    0x3d800000u, 0x3d900000u, 0x3da00000u, 0x3db00000u, 0x3dc00000u, 0x3dd00000u, 0x3de00000u, 0x3df00000u,
    0x3e000000u, 0x3e100000u, 0x3e200000u, 0x3e300000u, 0x3e400000u, 0x3e500000u, 0x3e600000u, 0x3e700000u,
    0x3e800000u, 0x3e900000u, 0x3ea00000u, 0x3eb00000u, 0x3ec00000u, 0x3ed00000u, 0x3ee00000u, 0x3ef00000u,
    0x3f000000u, 0x3f100000u, 0x3f200000u, 0x3f300000u, 0x3f400000u, 0x3f500000u, 0x3f600000u, 0x3f700000u,
    0x3f800000u, 0x3f900000u, 0x3fa00000u, 0x3fb00000u, 0x3fc00000u, 0x3fd00000u, 0x3fe00000u, 0x3ff00000u,
    0x40000000u, 0x40100000u, 0x40200000u, 0x40300000u, 0x40400000u, 0x40500000u, 0x40600000u, 0x40700000u,
    0x40800000u, 0x40900000u, 0x40a00000u, 0x40b00000u, 0x40c00000u, 0x40d00000u, 0x40e00000u, 0x40f00000u,
    0x41000000u, 0x41100000u, 0x41200000u, 0x41300000u, 0x41400000u, 0x41500000u, 0x41600000u, 0x41700000u,
    0x41800000u, 0x41900000u, 0x41a00000u, 0x41b00000u, 0x41c00000u, 0x41d00000u, 0x41e00000u, 0x41f00000u,
    0x42000000u, 0x42100000u, 0x42200000u, 0x42300000u, 0x42400000u, 0x42500000u, 0x42600000u, 0x42700000u,
    0x42800000u, 0x42900000u, 0x42a00000u, 0x42b00000u, 0x42c00000u, 0x42d00000u, 0x42e00000u, 0x42f00000u,
    0x43000000u, 0x43100000u, 0x43200000u, 0x43300000u, 0x43400000u, 0x43500000u, 0x43600000u, 0x43700000u,
    0x7f800000u, 0x7fc00000u, 0x7fc00000u, 0x7fc00000u, 0x7fc00000u, 0x7fc00000u, 0x7fc00000u, 0x7fc00000u,
    0x80000000u, 0xbb000000u, 0xbb800000u, 0xbbc00000u, 0xbc000000u, 0xbc200000u, 0xbc400000u, 0xbc600000u,
    0xbc800000u, 0xbc900000u, 0xbca00000u, 0xbcb00000u, 0xbcc00000u, 0xbcd00000u, 0xbce00000u, 0xbcf00000u,
    0xbd000000u, 0xbd100000u, 0xbd200000u, 0xbd300000u, 0xbd400000u, 0xbd500000u, 0xbd600000u, 0xbd700000u,
    0xbd800000u, 0xbd900000u, 0xbda00000u, 0xbdb00000u, 0xbdc00000u, 0xbdd00000u, 0xbde00000u, 0xbdf00000u,
    0xbe000000u, 0xbe100000u, 0xbe200000u, 0xbe300000u, 0xbe400000u, 0xbe500000u, 0xbe600000u, 0xbe700000u,
    0xbe800000u, 0xbe900000u, 0xbea00000u, 0xbeb00000u, 0xbec00000u, 0xbed00000u, 0xbee00000u, 0xbef00000u,
    0xbf000000u, 0xbf100000u, 0xbf200000u, 0xbf300000u, 0xbf400000u, 0xbf500000u, 0xbf600000u, 0xbf700000u,
    0xbf800000u, 0xbf900000u, 0xbfa00000u, 0xbfb00000u, 0xbfc00000u, 0xbfd00000u, 0xbfe00000u, 0xbff00000u,
    0xc0000000u, 0xc0100000u, 0xc0200000u, 0xc0300000u, 0xc0400000u, 0xc0500000u, 0xc0600000u, 0xc0700000u,
    0xc0800000u, 0xc0900000u, 0xc0a00000u, 0xc0b00000u, 0xc0c00000u, 0xc0d00000u, 0xc0e00000u, 0xc0f00000u,
    0xc1000000u, 0xc1100000u, 0xc1200000u, 0xc1300000u, 0xc1400000u, 0xc1500000u, 0xc1600000u, 0xc1700000u,
    0xc1800000u, 0xc1900000u, 0xc1a00000u, 0xc1b00000u, 0xc1c00000u, 0xc1d00000u, 0xc1e00000u, 0xc1f00000u,
    0xc2000000u, 0xc2100000u, 0xc2200000u, 0xc2300000u, 0xc2400000u, 0xc2500000u, 0xc2600000u, 0xc2700000u,
    0xc2800000u, 0xc2900000u, 0xc2a00000u, 0xc2b00000u, 0xc2c00000u, 0xc2d00000u, 0xc2e00000u, 0xc2f00000u,
    0xc3000000u, 0xc3100000u, 0xc3200000u, 0xc3300000u, 0xc3400000u, 0xc3500000u, 0xc3600000u, 0xc3700000u,
    0xff800000u, 0xffc00000u, 0xffc00000u, 0xffc00000u, 0xffc00000u, 0xffc00000u, 0xffc00000u, 0xffc00000u};

inline float decode_e2m1(std::uint8_t raw) {
  constexpr int kExponentBits = 2;
  constexpr int kMantissaBits = 1;
  constexpr int kBias = (1 << (kExponentBits - 1)) - 1;
  constexpr int kMaxExponent = (1 << kExponentBits) - 1;

  const bool sign = ((raw >> (kExponentBits + kMantissaBits)) & 1u) != 0u;
  const auto exponent = (raw >> kMantissaBits) & ((1 << kExponentBits) - 1);
  const auto mantissa = raw & ((1 << kMantissaBits) - 1);

  if (exponent == 0) {
    if (mantissa == 0) {
      return std::bit_cast<float>(canonical_zero_bits(sign));
    }
    const float fraction = static_cast<float>(mantissa) / static_cast<float>(1 << kMantissaBits);
    const float value = std::ldexp(fraction, 1 - kBias);
    return sign ? -value : value;
  }

  if (exponent == kMaxExponent) {
    return std::bit_cast<float>(mantissa == 0 ? canonical_inf_bits(sign) : canonical_nan_bits(sign));
  }

  const float fraction = static_cast<float>(mantissa) / static_cast<float>(1 << kMantissaBits);
  const float value = std::ldexp(1.0f + fraction, exponent - kBias);
  return sign ? -value : value;
}

inline float decode_ue4m3(std::uint8_t raw) {
  constexpr int kExponentBits = 4;
  constexpr int kMantissaBits = 3;
  constexpr int kBias = (1 << (kExponentBits - 1)) - 1;
  constexpr int kMaxExponent = (1 << kExponentBits) - 1;

  const bool sign = ((raw >> (kExponentBits + kMantissaBits)) & 1u) != 0u;
  const auto exponent = (raw >> kMantissaBits) & ((1 << kExponentBits) - 1);
  const auto mantissa = raw & ((1 << kMantissaBits) - 1);

  if (exponent == 0) {
    if (mantissa == 0) {
      return std::bit_cast<float>(canonical_zero_bits(sign));
    }
    const float fraction = static_cast<float>(mantissa) / static_cast<float>(1 << kMantissaBits);
    const float value = std::ldexp(fraction, 1 - kBias);
    return sign ? -value : value;
  }

  if (exponent == kMaxExponent) {
    return std::bit_cast<float>(mantissa == 0 ? canonical_inf_bits(sign) : canonical_nan_bits(sign));
  }

  const float fraction = static_cast<float>(mantissa) / static_cast<float>(1 << kMantissaBits);
  const float value = std::ldexp(1.0f + fraction, exponent - kBias);
  return sign ? -value : value;
}

}  // namespace sm120::unit
