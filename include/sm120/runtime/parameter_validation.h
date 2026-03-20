#pragma once

#include <cstddef>
#include <string>

namespace sm120::runtime {

enum class DataType {
  Float16,
  Float32,
};

struct LaunchConfig {
  int batch_size = 1;
  int sequence_length = 1;
  int hidden_size = 128;
  int head_count = 1;
  int pipeline_width = 1;
  bool use_bias = false;
  DataType dtype = DataType::Float16;
};

struct ValidationResult {
  bool success;
  std::string message;
};

constexpr int kMaxBatchSize = 512;
constexpr int kMinBatchSize = 1;
constexpr int kMaxSequenceLength = 16384;
constexpr int kMinSequenceLength = 16;
constexpr int kMaxHiddenSize = 32768;
constexpr int kMinHiddenSize = 64;
constexpr int kHiddenAlign = 32;
constexpr int kMaxHeadCount = 64;

ValidationResult validate_launch_config(LaunchConfig const& config);

}  // namespace sm120::runtime
