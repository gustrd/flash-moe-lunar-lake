# 🚩 PROJECT MIGRATED & PIVOTED

This repository is no longer the active home of development. All work has migrated to the **`vendor/llama.cpp`** fork.

## 🔄 The Pivot: Two-Instance → Single-Process

The original design used a two-instance Python orchestrator. This approach was **abandoned** due to a fatal flaw:
- **Garbage Propagation**: In an MoE model, routing decisions for layer `L+1` depend on the MoE output of layer `L`. 
- By using "stub" (zero) weights in the router instance, the MoE output was garbage, leading to nonsensical routing in all subsequent layers.
- The only way to get correct routing is to have **real weights** loaded at the moment of inference for every layer.

## 🚀 New Architecture: In-Graph Expert Hook

The engine now lives **entirely inside C++** within the `koboldcpp` (llama.cpp) fork.

| Feature | New Implementation |
|---|---|
| **Language** | Pure C++ (with original Python tools for offline extraction) |
| **Logic** | Hooks into `ggml-backend.cpp` selective expert copy path |
| **I/O** | Native Win32 `ReadFile` with `FILE_FLAG_NO_BUFFERING` |
| **Cache** | Native C++ LRU (`std::unordered_map` + linked list) |
| **Process** | Single process (no IPC, no Python orchestrator needed) |

## 📂 Where is the code now?

All active files are located in the `vendor/llama.cpp` submodule on the `flash-moe` branch:

1.  **New Plan**: [vendor/llama.cpp/PLAN.md](file:///c:/Users/gustr/_git/flash-moe-lunar-lake/vendor/llama.cpp/PLAN.md)
2.  **C++ Source**: `vendor/llama.cpp/src/` (patches for loader, backend, and new flash_moe files)
3.  **Python Tools**: [vendor/llama.cpp/flash-moe/](file:///c:/Users/gustr/_git/flash-moe-lunar-lake/vendor/llama.cpp/flash-moe/) (extracted from root for vertical integration)
4.  **Tests**: [vendor/llama.cpp/flash-moe/tests/](file:///c:/Users/gustr/_git/flash-moe-lunar-lake/vendor/llama.cpp/flash-moe/tests/)

## 🛠️ How to continue

1.  Navigate to `vendor/llama.cpp`.
2.  Follow the instructions in the **new** `PLAN.md` located there.
3.  Build using `w64devkit` with `make LLAMA_VULKAN=1`.

---
*This root directory is preserved only as a container for the submodule. The Python files in this root are legacy and have been moved to the fork.*
