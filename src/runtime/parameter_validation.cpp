#include "sm120/runtime/parameter_validation.h"

#include <utility>

namespace sm120::runtime {

namespace {

ValidationResult make_failure(std::string message) {
  return {false, std::move(message)};
}

const char* to_string(DataType dtype) {
  return dtype == DataType::Float32 ? "float32" : "float16";
}

}  // namespace

ValidationResult validate_launch_config(LaunchConfig const& config) {
  if (config.batch_size < kMinBatchSize) {
    return make_failure("batch_size must be >= " + std::to_string(kMinBatchSize));
  }
  if (config.batch_size > kMaxBatchSize) {
    return make_failure("batch_size must be <= " + std::to_string(kMaxBatchSize));
  }

  if (config.sequence_length < kMinSequenceLength) {
    return make_failure("sequence_length must be >= " + std::to_string(kMinSequenceLength));
  }
  if (config.sequence_length > kMaxSequenceLength) {
    return make_failure("sequence_length must be <= " + std::to_string(kMaxSequenceLength));
  }

  if (config.hidden_size < kMinHiddenSize) {
    return make_failure("hidden_size must be >= " + std::to_string(kMinHiddenSize));
  }
  if (config.hidden_size > kMaxHiddenSize) {
    return make_failure("hidden_size must be <= " + std::to_string(kMaxHiddenSize));
  }
  if (config.hidden_size % kHiddenAlign != 0) {
    return make_failure("hidden_size must be a multiple of " + std::to_string(kHiddenAlign));
  }

  if (config.head_count < 1) {
    return make_failure("head_count must be >= 1");
  }
  if (config.head_count > kMaxHeadCount) {
    return make_failure("head_count must be <= " + std::to_string(kMaxHeadCount));
  }
  if (config.hidden_size % config.head_count != 0) {
    return make_failure("hidden_size must be divisible by head_count");
  }

  const int per_head = config.hidden_size / config.head_count;
  if (per_head < kHiddenAlign) {
    return make_failure("hidden_size / head_count must be >= " + std::to_string(kHiddenAlign));
  }

  if (config.pipeline_width < 1) {
    return make_failure("pipeline_width must be >= 1");
  }
  if (config.pipeline_width > config.sequence_length) {
    return make_failure("pipeline_width must not exceed sequence_length");
  }

  return {true, "configuration valid for " + std::string(to_string(config.dtype))};
}

}  // namespace sm120::runtime
