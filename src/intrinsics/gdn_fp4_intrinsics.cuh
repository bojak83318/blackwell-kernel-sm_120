#pragma once

#include <cmath>
#include <cstdint>
#include <limits>

namespace sm120::intrinsics {

namespace detail {

constexpr int kE2M1Bias = 1;
constexpr int kE2M1MantissaBits = 1;
constexpr int kE4M3Bias = 7;
constexpr int kE4M3MantissaBits = 3;
constexpr int kE4M3ExponentBits = 4;
constexpr int kE4M3MaxExponent = (1 << kE4M3ExponentBits) - 2;
constexpr int kE4M3MantissaMask = (1 << kE4M3MantissaBits) - 1;
constexpr float kE4M3MaxNormalized = 1.0f + static_cast<float>(kE4M3MantissaMask) / (1 << kE4M3MantissaBits);

inline __host__ __device__ float apply_sign(float magnitude, bool sign) {
  return sign ? -magnitude : magnitude;
}

}  // namespace detail

/// Decode an NVFP4 E2M1 nibble (4 bits) into a float32 value.
inline __host__ __device__ float decode_e2m1_to_f32(std::uint8_t value) noexcept {
  const std::uint8_t bits = value & 0xFu;
  const bool sign = (bits & 0x8u) != 0;
  const std::uint8_t exponent = (bits >> 1) & 0x3u;
  const std::uint8_t mantissa = bits & 0x1u;

  float magnitude = 0.0f;
  if (exponent == 0) {
    if (mantissa != 0) {
      magnitude = std::ldexp(static_cast<float>(mantissa) / (1u << detail::kE2M1MantissaBits),
                             1 - detail::kE2M1Bias);
    }
  } else {
    magnitude = std::ldexp(1.0f + static_cast<float>(mantissa) / (1u << detail::kE2M1MantissaBits),
                           exponent - detail::kE2M1Bias);
  }

  return detail::apply_sign(magnitude, sign);
}

/// Decode an unsigned E4M3 block scale into float32.
inline __host__ __device__ float decode_ue4m3_to_f32(std::uint8_t value) noexcept {
  const bool sign = (value & 0x80u) != 0;
  const std::uint8_t exponent = (value >> 3) & 0x0Fu;
  const std::uint8_t mantissa = value & detail::kE4M3MantissaMask;

  float magnitude;
  if (exponent == 0) {
    magnitude = (mantissa == 0)
                    ? 0.0f
                    : std::ldexp(static_cast<float>(mantissa) / (1u << detail::kE4M3MantissaBits),
                                 1 - detail::kE4M3Bias);
  } else if (exponent == 0xFu) {
    magnitude = (mantissa == 0) ? std::numeric_limits<float>::infinity()
                                : std::numeric_limits<float>::quiet_NaN();
  } else {
    magnitude = std::ldexp(1.0f + static_cast<float>(mantissa) / (1u << detail::kE4M3MantissaBits),
                           exponent - detail::kE4M3Bias);
  }

  return detail::apply_sign(magnitude, sign);
}

/// Encode a positive float32 value into the unsigned E4M3 layout.
inline __host__ __device__ std::uint8_t encode_f32_to_ue4m3(float value) noexcept {
  if (!std::isfinite(value) || value <= 0.0f) {
    return 0;
  }

  const float max_value = detail::kE4M3MaxNormalized * std::ldexp(1.0f, detail::kE4M3MaxExponent - detail::kE4M3Bias);
  if (value >= max_value) {
    return static_cast<std::uint8_t>((detail::kE4M3MaxExponent << detail::kE4M3MantissaBits) |
                                      detail::kE4M3MantissaMask);
  }

  const float min_normal = std::ldexp(1.0f, 1 - detail::kE4M3Bias);
  if (value < min_normal) {
    const float unit = std::ldexp(1.0f, 1 - detail::kE4M3Bias - detail::kE4M3MantissaBits);
    int mantissa = static_cast<int>(value / unit + 0.5f);
    if (mantissa <= 0) {
      return 0;
    }
    if (mantissa > detail::kE4M3MantissaMask) {
      mantissa = detail::kE4M3MantissaMask;
    }
    return static_cast<std::uint8_t>(mantissa);
  }

  int unbiased_exponent = std::ilogb(value);
  float normalized = std::ldexp(value, -unbiased_exponent);
  if (normalized >= detail::kE4M3MaxNormalized) {
    normalized *= 0.5f;
    ++unbiased_exponent;
  }

  int exponent_field = unbiased_exponent + detail::kE4M3Bias;
  if (exponent_field > detail::kE4M3MaxExponent) {
    exponent_field = detail::kE4M3MaxExponent;
    normalized = detail::kE4M3MaxNormalized;
  }
  if (exponent_field <= 0) {
    return 0;
  }

  float mantissa_scaled = (normalized - 1.0f) * (1u << detail::kE4M3MantissaBits);
  int mantissa = static_cast<int>(mantissa_scaled + 0.5f);
  if (mantissa > detail::kE4M3MantissaMask) {
    mantissa = 0;
    ++exponent_field;
    if (exponent_field > detail::kE4M3MaxExponent) {
      exponent_field = detail::kE4M3MaxExponent;
      mantissa = detail::kE4M3MantissaMask;
    }
  }

  return static_cast<std::uint8_t>((exponent_field << detail::kE4M3MantissaBits) | mantissa);
}

}  // namespace sm120::intrinsics
