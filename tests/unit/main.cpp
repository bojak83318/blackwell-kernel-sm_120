#include <iostream>
#include <string>
#include <string_view>

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

  if (list_only) {
    std::cout << "SM_120 unit harness placeholder: available tests -> [";
    std::cout << "context_check, resource_setup";
    std::cout << "]\n";
    return 0;
  }

  std::cout << "SM_120 unit test harness running in '" << mode << "' mode";
  std::cout << (verbose ? " (verbose)" : "") << "\n";
  std::cout << "-> placeholder setup complete, no kernels executed yet" << std::endl;
  return 0;
}
