#include "sm120/runtime/host_launcher.h"

#include <cstdint>

namespace sm120::runtime {

namespace {

ValidationResult make_failure(std::string message) {
  return {false, std::move(message)};
}

constexpr std::size_t kBufferAlignment = 128;
constexpr std::size_t kMinWorkspaceBytes = 4096;

bool is_aligned(const void* ptr) {
  auto value = reinterpret_cast<std::uintptr_t>(ptr);
  return (value % kBufferAlignment) == 0;
}

ValidationResult validate_buffers(HostLaunchContext const& context) {
  if (context.buffers.gdn_input == nullptr) {
    return make_failure("gdn_input buffer cannot be null");
  }
  if (context.buffers.gdn_output == nullptr) {
    return make_failure("gdn_output buffer cannot be null");
  }
  if (context.workspace_bytes < kMinWorkspaceBytes) {
    return make_failure("workspace_bytes must be >= " + std::to_string(kMinWorkspaceBytes));
  }
  if (!is_aligned(context.buffers.gdn_input)) {
    return make_failure("gdn_input must be aligned to " + std::to_string(kBufferAlignment) + " bytes");
  }
  if (!is_aligned(context.buffers.gdn_output)) {
    return make_failure("gdn_output must be aligned to " + std::to_string(kBufferAlignment) + " bytes");
  }
  if (context.buffers.bias && !is_aligned(context.buffers.bias)) {
    return make_failure("bias buffer must be aligned to " + std::to_string(kBufferAlignment) + " bytes");
  }
  return {true, "buffers aligned"};
}

}  // namespace

LaunchResult dispatch_gdn_launch(HostLaunchContext const& context) {
  auto validation = validate_launch_config(context.config);
  if (!validation.success) {
    return {false, "config validation failed: " + validation.message};
  }

  auto buffer_validation = validate_buffers(context);
  if (!buffer_validation.success) {
    return {false, "buffer validation failed: " + buffer_validation.message};
  }

  std::string dtype_desc = context.config.dtype == DataType::Float32 ? "float32" : "float16";
  std::string message = "dispatch ready for dtype=" + dtype_desc + " seq=" +
      std::to_string(context.config.sequence_length);
  return {true, message};
}

}  // namespace sm120::runtime
