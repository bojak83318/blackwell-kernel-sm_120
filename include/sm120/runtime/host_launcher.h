#pragma once

#include "sm120/runtime/parameter_validation.h"

#include <cstddef>
#include <string>

namespace sm120::runtime {

struct LaunchBuffers {
  const void* gdn_input = nullptr;
  void* gdn_output = nullptr;
  const void* bias = nullptr;
};

struct HostLaunchContext {
  LaunchConfig config;
  LaunchBuffers buffers;
  std::size_t workspace_bytes = 0;
};

struct LaunchResult {
  bool success;
  std::string message;
};

LaunchResult dispatch_gdn_launch(HostLaunchContext const& context);

}  // namespace sm120::runtime
