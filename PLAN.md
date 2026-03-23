# Flash-MoE on Lunar Lake: Expert Streaming Engine
## Execution Plan for Claude Code

> **Hardware**: Intel Lunar Lake, 32 GB LPDDR5X on-package, Arc 140V iGPU (Xe2, 64 EUs)
> **Model**: Qwen3-30B-A3B (30.5B total, 3.3B active, 128 experts/layer, K=8, 48 layers)
> **Backend**: llama.cpp with Vulkan (confirmed working)
> **OS**: Windows 11
> **Constraint**: Zero swap writes. All SSD I/O must be explicit, read-only, controlled by our code.
> **Goal**: Prove the flash-moe SSD expert streaming technique on x86/Windows/Vulkan, then scale to DeepSeek V3.2.
> **Language**: Python is the primary implementation language for all tools, engine logic, and orchestration. C is used only where Windows API calls are unavoidable and cannot be driven from Python (e.g. `FILE_FLAG_NO_BUFFERING` via ctypes). Prefer ctypes/cffi over writing new C source files.
> **llama.cpp**: Managed as a git submodule (`vendor/llama.cpp`). Use official pre-built release binaries (downloaded via script) wherever possible. Compile from source only when a required feature is missing from the release build.
> **Input model**: Must be a pre-quantized `.gguf` file (e.g. `Qwen3-30B-A3B-Q4_K_M.gguf`). We do **not** quantize models ourselves. The user downloads the GGUF from HuggingFace (e.g. bartowski/Qwen_Qwen3-30B-A3B-GGUF) and provides its path. No float16/bfloat16 safetensors → GGUF conversion is in scope.

---

## Table of Contents

1. [Background and Architecture](#1-background-and-architecture)
2. [Memory Budget](#2-memory-budget)
3. [Phase 2A — Expert Extraction Tool](#3-phase-2a--expert-extraction-tool)
4. [Phase 2B — Streaming Engine Core](#4-phase-2b--streaming-engine-core)
5. [Phase 2C — Integration with llama.cpp](#5-phase-2c--integration-with-llamacpp)
6. [Phase 2D — Benchmarking Suite](#6-phase-2d--benchmarking-suite)
7. [Phase 3 — Optimizations](#7-phase-3--optimizations)
8. [Scaling Path to DeepSeek V3.2](#8-scaling-path-to-deepseek-v32)
9. [Reference Materials](#9-reference-materials)

---

## 1. Background and Architecture

### 1.1 What flash-moe does

The [danveloper/flash-moe](https://github.com/danveloper/flash-moe) project runs Qwen3.5-397B-A17B on a 48 GB MacBook Pro M3 Max at 5.5 tok/s by streaming MoE expert weights from NVMe SSD on demand. Core techniques:

- Expert weights live on SSD (~120 GB at 2-bit). Only the K=4 active experts per layer are read per token.
- Non-expert weights (attention, embeddings, routing, norms) stay permanently in RAM (~5.5 GB).
- Parallel `pread()` with `F_NOCACHE` bypasses OS page cache for direct SSD reads.
- Metal GPU shaders handle dequantization and matrix math.
- Pipeline overlaps I/O of layer N+1 with compute of layer N.

### 1.2 What we are building

A Windows/Vulkan equivalent of the flash-moe engine, targeting Qwen3-30B-A3B as proof of concept:

```
┌─────────────────────────────────────────────────────────┐
│                    Inference Pipeline                     │
│                                                           │
│  ┌──────────┐    ┌──────────┐    ┌───────────────────┐   │
│  │ model_base│    │  Expert  │    │   Expert I/O      │   │
│  │  (RAM)    │◄──►│  Cache   │◄──►│   Pool (SSD)      │   │
│  │  ~6 GB    │    │  (LRU)   │    │  FILE_FLAG_NO_    │   │
│  │           │    │  ≤19 GB  │    │  BUFFERING         │   │
│  └──────────┘    └──────────┘    └───────────────────┘   │
│       │                                                    │
│       ▼                                                    │
│  ┌──────────┐    ┌──────────┐    ┌───────────────────┐   │
│  │  Vulkan   │    │   CPU    │    │    Output         │   │
│  │  (Attn,   │◄──►│  (Expert │───►│    Sampling       │   │
│  │   Norms)  │    │  MatMul) │    │                   │   │
│  └──────────┘    └──────────┘    └───────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 1.3 Qwen3-30B-A3B architecture summary

| Property | Value |
|---|---|
| Total parameters | 30.5B |
| Active parameters/token | 3.3B |
| Transformer layers | 48 |
| Hidden dimension | 2048 |
| Attention | GQA, 32 Q-heads, 4 KV-heads |
| Experts per MoE layer | 128 routed |
| Active experts per token (K) | 8 |
| Expert FFN intermediate | model-specific (check config.json) |
| Context window | 32K native, 131K via YaRN |
| GGUF size (Q4_K_M) | ~18-19 GB |

### 1.4 Why this model for the PoC

- **It fits in 32 GB RAM entirely.** This lets us compare streaming vs fully-loaded baselines on identical hardware.
- **Small experts (~250-500 KB each).** Total expert payload per token is ~192 MB (48 layers × 8 experts × ~500 KB), giving ~32 ms I/O overhead at 6 GB/s — fast enough to be interactive.
- **Standard MoE architecture.** GQA attention + SwiGLU experts + top-K routing. No exotic components like MLA or GatedDeltaNet. The streaming technique is isolatable.
- **llama.cpp has full support** for this architecture, including Vulkan backend and MoE expert CPU offloading.

---

## 2. Memory Budget

### 2.1 Hard limits

```
Total physical RAM:                          32,768 MB
Windows + drivers + services:                ~4,096 MB
Pagefile (fixed, for crash dumps only):       1,024 MB
────────────────────────────────────────────────────────
Available for engine:                       ~27,648 MB

Non-expert weights (attention, embed, norms):  ~6,000 MB
KV cache (4K context, Q8_0):                    ~512 MB
Vulkan scratch buffers:                         ~512 MB
────────────────────────────────────────────────────────
Fixed engine consumption:                    ~7,024 MB

Available for expert cache:                 ~20,624 MB
Expert I/O work buffers (K=8 × ~512 KB):         ~4 MB
```

### 2.2 Expert sizing (verify in Phase 2A)

Each expert consists of three matrices (gate_proj, up_proj, down_proj) for SwiGLU.
At Q4_K_M quantization, estimated size per expert: **~250-500 KB** (to be measured precisely).
Total experts: 128 × 48 = 6,144.
Total expert weight volume: **~12-13 GB** (to be measured precisely).

With ~20 GB of cache space, **all experts fit in RAM**. The PoC validates the streaming mechanism; scaling tests artificially limit cache size to simulate larger models.

### 2.3 Zero-swap enforcement

Three layers of defense, in order of importance:

1. **Pagefile fixed at 1 GB**: Prevents Windows from auto-expanding the pagefile. Settings: `System Properties → Advanced → Performance Settings → Advanced → Virtual Memory → Custom size: 1024/1024`.

2. **`--no-mmap` + `--mlock`**: Forces llama.cpp to read weights into heap memory (not memory-mapped file pages) and pin them in the working set via `VirtualLock()`.

3. **`SetProcessWorkingSetSize(min=20GB, max=26GB)`**: Tells the Windows VMM to keep at least 20 GB in the process working set, preventing aggressive page trimming.

4. **All expert data is read-only**: Cache eviction never writes back to SSD. It simply calls `VirtualFree()`.

**Caveat**: `VirtualLock` on Windows locks pages into the *working set*, not into physical RAM unconditionally. When all threads are blocked, pages may still be paged out. Raymond Chen documented this at https://devblogs.microsoft.com/oldnewthing/20071106-00/?p=24573. In practice, during active inference with continuous GPU/CPU work, the working set stays resident.

---

## 3. Phase 2A — Expert Extraction Tool

### 3.1 Objective

Create a Python tool that splits a **pre-quantized GGUF file** (provided by the user, e.g. `Qwen3-30B-A3B-Q4_K_M.gguf` downloaded from HuggingFace) into:
- `model_base.gguf` — All non-expert tensors (attention, embeddings, norms, routing weights)
- `experts/` directory — One binary file per expert per layer, 4K-aligned, ready for direct I/O

**No quantization step**: the input GGUF is already quantized (Q4_K_M, Q2_K, etc.). We extract tensors verbatim — their quantization format is preserved as-is.

### 3.2 Prerequisites

```bash
pip install gguf numpy
```

Verify GGUF can be read:
```bash
python -c "import gguf; print(gguf.__version__)"
```

### 3.3 Step-by-step

#### Step 1: Inspect the GGUF tensor layout

```bash
# Script: inspect_gguf.py
# Purpose: List all tensors, their shapes, sizes, and identify expert tensors
# Output: JSON manifest of tensor names, shapes, dtypes, byte offsets, byte sizes
# Flag expert tensors (contain "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")
```

- [ ] Parse the GGUF header and tensor metadata
- [ ] For each tensor, record: name, shape, dtype (quantization type), offset, byte size
- [ ] Identify the naming convention: how layer index and expert index are encoded
- [ ] Determine whether experts are stored as one big tensor per layer (shape `[n_experts, ...]`) or as individual tensors per expert
- [ ] Output a JSON manifest: `gguf_manifest.json`

**Test: `test_inspect_gguf.py`**
```
- [ ] Manifest contains exactly 48 layers worth of expert tensors
- [ ] Each expert tensor group (gate, up, down) has consistent shapes
- [ ] Total byte count of expert tensors matches expected ~12-13 GB
- [ ] Non-expert tensor count matches expected (attn, embed, norm, router)
- [ ] All tensor offsets are valid (within file bounds)
```

#### Step 2: Extract experts to individual files

```bash
# Script: extract_experts.py
# Input: Original GGUF file + manifest from Step 1
# Output: experts/ directory with one file per expert
# Naming: blk{LL}_exp{EEE}.bin (e.g., blk00_exp000.bin)
# Each file contains gate+up+down weights concatenated, 4K-aligned
```

- [ ] Read each expert's gate, up, down tensors from the GGUF
- [ ] Concatenate into a single contiguous buffer per expert
- [ ] Pad to next 4096-byte boundary (required for `FILE_FLAG_NO_BUFFERING`)
- [ ] Write to `experts/blk{LL}_exp{EEE}.bin`
- [ ] Generate `expert_index.json`: for each (layer, expert_id), record file path, byte size, gate/up/down offsets within the file, quantization type, original tensor shapes

**Test: `test_extract_experts.py`**
```
- [ ] Number of files == 48 × 128 == 6,144
- [ ] Every file size is a multiple of 4096
- [ ] File sizes are consistent within the same layer (all experts same size)
- [ ] expert_index.json is valid JSON with 6,144 entries
- [ ] Byte-for-byte comparison: reading expert data from original GGUF matches the extracted file content (sample 10 random experts)
- [ ] Total size of experts/ directory matches expected ~12-13 GB
```

#### Step 3: Create base GGUF without experts

```bash
# Script: create_base_gguf.py
# Input: Original GGUF + manifest
# Output: model_base.gguf containing only non-expert tensors
# Expert tensor entries are either removed or replaced with zero-byte stubs
```

- [ ] Copy all GGUF metadata (model architecture, tokenizer, hyperparameters)
- [ ] Copy all non-expert tensors verbatim
- [ ] For expert tensors: either omit entirely or write a placeholder with correct metadata but zero data
- [ ] Write valid GGUF to `model_base.gguf`

**Test: `test_create_base_gguf.py`**
```
- [ ] model_base.gguf is a valid GGUF file (parseable by gguf-py)
- [ ] Size is approximately 5-6 GB (non-expert weights only)
- [ ] Contains all expected non-expert tensors with correct shapes and dtypes
- [ ] Does NOT contain expert weight data
- [ ] Tokenizer metadata is preserved (vocab size, special tokens)
```

#### Integration test for Phase 2A

```
- [ ] Round-trip validation: reconstruct full model data by combining model_base.gguf + experts/ 
      and verify byte-for-byte match against the original GGUF for a sample of tensors
- [ ] expert_index.json enables O(1) lookup: given (layer=23, expert_id=45), 
      can compute file path and internal offsets in constant time
```

---

## 4. Phase 2B — Streaming Engine Core

### 4.1 Objective

Implement three modules in Python, tested independently:
- **Expert Cache** (LRU, fixed memory budget) — pure Python with `mmap`/`bytearray` backing
- **Expert I/O Pool** (async Windows file reads, 4K-aligned) — Python calling Windows API via `ctypes`
- **Expert Dispatcher** (routing decision → cache lookup → I/O dispatch → compute handoff) — pure Python

**Language policy**: All modules are Python. The I/O pool calls `CreateFile`, `ReadFile`, `VirtualAlloc`, and `WaitForMultipleObjects` through `ctypes.windll` / `ctypes.WinDLL` — no separate C source files unless a specific call is proven impossible from Python (document the reason if so).

### 4.2 Prerequisites

```bash
pip install numpy
# Python 3.11+ (ctypes.windll available on Windows by default)
# No CMake, no C compiler required
```

### 4.3 Module 1: Expert Cache

```
File: expert_cache.py
```

Data structures:
- `OrderedDict`: `(layer, expert_id) → bytes` — O(1) lookup with LRU promotion via `move_to_end`
- Explicit `used_bytes` counter; evict from front when over budget

API:
```python
class ExpertCache:
    def __init__(self, max_bytes: int): ...
    def get(self, layer: int, expert_id: int) -> bytes | None: ...
    def put(self, layer: int, expert_id: int, data: bytes) -> None: ...
    def clear(self) -> None: ...
    @property
    def used_bytes(self) -> int: ...
    @property
    def hit_rate(self) -> float: ...
    @property
    def hits(self) -> int: ...
    @property
    def misses(self) -> int: ...
```

Implementation notes:
- `put` evicts LRU entries until space is available. Eviction simply removes from the dict — **never writes to disk**.
- Key is `(layer, expert_id)` tuple; total unique keys = 48 × 128 = 6,144 — trivially small for a dict.
- Data stored as `bytes` objects (immutable, ref-counted). No manual memory management needed.

- [ ] Implement `expert_cache_create` / `expert_cache_destroy`
- [ ] Implement `expert_cache_get` (LRU promotion on hit)
- [ ] Implement `expert_cache_put` (LRU eviction, VirtualAlloc for data)
- [ ] Implement `expert_cache_clear`
- [ ] Implement hit/miss counters and hit rate

**Test: `test_expert_cache.c`**
```
- [ ] Put + get roundtrip: data integrity for 1, 10, 100 experts
- [ ] LRU eviction order: insert A, B, C with max_bytes fitting only 2. 
      Verify A is evicted when D is inserted.
- [ ] LRU promotion: insert A, B, C (max fits 2). Get A. Insert D. 
      Verify B is evicted (not A, because A was accessed).
- [ ] Memory limit respected: used_bytes never exceeds max_bytes
- [ ] Hit rate accuracy: 50 puts, 50 gets (25 hits, 25 misses) → hit_rate == 0.5
- [ ] Clear resets all state: used == 0, hit_rate == 0
- [ ] Stress test: 10,000 random put/get operations, verify no memory leak 
      (check with _CrtDumpMemoryLeaks on MSVC or equivalent)
- [ ] Zero writes to disk during eviction: 
      monitor with a file system watcher or performance counter
```

### 4.4 Module 2: Expert I/O Pool

```
File: expert_io.py
```

API:
```python
class ExpertIOPool:
    def __init__(self, experts_dir: str, index_json_path: str): ...

    # Submit K read requests. Non-blocking. Returns immediately.
    def submit(self, layer: int, expert_ids: list[int]) -> None: ...

    # Wait for all submitted reads. Returns list of bytes objects.
    def wait(self) -> list[bytes]: ...

    @property
    def avg_latency_ms(self) -> float: ...
    @property
    def total_bytes_read(self) -> int: ...
```

Implementation notes:
- Uses `ctypes.windll.kernel32` to call `CreateFileW` with `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED`.
- Allocates 4K-aligned read buffers via `ctypes.windll.kernel32.VirtualAlloc`.
- Issues overlapped reads with `OVERLAPPED` structs created in Python via `ctypes.Structure`.
- `wait()` calls `WaitForMultipleObjects` via ctypes.
- Reads are rounded up to the next 4096-byte boundary (required by `FILE_FLAG_NO_BUFFERING`).
- Loads `expert_index.json` at init to map `(layer, expert_id)` → file path and size.
- If any specific Windows call proves impossible from ctypes, isolate it in a minimal `io_helpers.c` shim (document why).

- [ ] Implement `expert_io_create`: load index, allocate buffers
- [ ] Implement `expert_io_submit`: open files, issue async reads
- [ ] Implement `expert_io_wait`: wait for completion, return buffers
- [ ] Implement handle pooling (optional optimization: keep handles open)
- [ ] Implement latency tracking (QueryPerformanceCounter around wait)

**Test: `test_expert_io.c`**
```
- [ ] Single expert read: submit 1, wait, verify data matches extracted file
- [ ] Batch read (K=8): submit 8, wait, all 8 correct
- [ ] Sequential reads across layers: read experts from layer 0, 1, ..., 47
- [ ] Concurrent batch: submit 8 from layer 0, then immediately 8 from layer 1 
      (verify no handle collision)
- [ ] Data integrity: byte-for-byte comparison of read buffer vs file content 
      for 100 random (layer, expert_id) pairs
- [ ] Alignment: verify all buffers are 4096-aligned (buffer_addr % 4096 == 0)
- [ ] Read size: verify actual read size is multiple of 4096 
      (even if expert is smaller, padding is read)
- [ ] Latency measurement: avg_latency_ms returns a positive number after reads
- [ ] Error handling: non-existent expert file returns error, does not crash
- [ ] Zero disk writes: monitor disk write counter before and after 1000 reads. 
      Delta must be zero for the experts/ volume.
```

### 4.5 Module 3: Expert Dispatcher

```
File: expert_dispatch.py
```

API:
```python
from dataclasses import dataclass

@dataclass
class DispatchStats:
    cache_hits: int
    cache_misses: int
    io_latency_ms: float

class ExpertDispatcher:
    def __init__(self, cache: ExpertCache, io: ExpertIOPool): ...

    # Given routing results (K expert IDs for a layer), return list of bytes.
    # Tries cache first. On miss, reads from SSD via I/O pool.
    def get(self, layer: int, expert_ids: list[int]) -> list[bytes]: ...

    def stats(self, layer: int) -> DispatchStats: ...
```

Implementation:
```
For each of the K expert_ids:
  1. Check cache: expert_cache_get(layer, expert_id)
  2. If hit: out_data[i] = cached pointer. Done.
  3. If miss: add to pending I/O batch.

If pending batch is non-empty:
  4. expert_io_submit(pool, layer, pending_ids, n_pending)
  5. expert_io_wait(pool, pending_buffers, pending_sizes, n_pending)
  6. For each returned buffer:
     a. expert_cache_put(layer, expert_id, buffer, size)  // cache for future
     b. out_data[i] = cached pointer (put returns the cache's own copy)
```

- [ ] Implement cache-first dispatch logic
- [ ] Implement I/O fallback for cache misses
- [ ] Implement per-layer stats tracking
- [ ] Handle mixed hits/misses within a single batch (some from cache, some from SSD)

**Test: `test_expert_dispatch.c`**
```
- [ ] All cache hits: pre-populate cache, dispatch K=8 experts, verify zero I/O
- [ ] All cache misses (cold start): empty cache, dispatch K=8, verify 8 reads
- [ ] Mixed: populate 4 of 8 experts in cache, dispatch 8, verify 4 reads
- [ ] Repeated dispatch same experts: first call = 8 misses, second call = 8 hits
- [ ] Cross-layer: dispatch layer 0 then layer 1. Layer 0 experts remain in cache.
- [ ] Eviction scenario: set cache to hold only 100 experts. 
      Dispatch 200 unique experts across layers. Verify LRU eviction and 
      re-reads work correctly.
- [ ] Stats accuracy: after known hit/miss pattern, verify stats match
- [ ] Data integrity: dispatched data must match original expert files
```

### Integration test for Phase 2B

```
- [ ] Full pipeline: Create cache (1 MB limit) + I/O pool + dispatcher.
      Simulate 100 tokens: for each token, for each of 48 layers, 
      dispatch K=8 random expert IDs.
      Verify: no crashes, data integrity, hit rate improves over time,
      total disk reads decrease as cache warms, zero disk writes.

- [ ] Memory bound: set cache to 100 MB. Run 1000-token simulation.
      Monitor process committed memory (GetProcessMemoryInfo). 
      Must never exceed cache_size + base_overhead + tolerance (50 MB).

- [ ] Performance: measure end-to-end dispatch latency per layer.
      With warm cache: < 1 ms per layer.
      Cold cache (SSD read): < 10 ms per layer.
```

---

## 5. Phase 2C — Integration with llama.cpp

### 5.1 Objective

Use the pre-built llama.cpp release binary to run inference while our Python streaming engine pre-populates the expert weights. The result is a working inference pipeline that:
- Loads `model_base.gguf` (non-expert weights) into RAM via `--no-mmap --mlock`
- Streams expert weights from `experts/` directory via the Python dispatcher
- Drives the llama.cpp subprocess from Python for full end-to-end control

### 5.2 llama.cpp setup: submodule + pre-built binaries

```
vendor/llama.cpp/   — git submodule (pinned to a known stable tag)
vendor/bin/         — pre-built Windows Vulkan release binaries (downloaded by setup.py)
```

**Policy**: Download the official GitHub release zip (e.g., `llama-<tag>-bin-win-vulkan-x64.zip`) via `setup.py`. Unzip into `vendor/bin/`. The submodule is present for source reference and as a fallback build path, but we do **not** compile by default. Only compile from source if a required feature is absent from the release build, and document the reason.

```
setup.py            — downloads correct llama.cpp release zip, verifies SHA256, extracts to vendor/bin/
```

### 5.3 Approach: Python wrapper around llama.cpp binary

Rather than patching C++ source, use llama.cpp's existing expert CPU-offload feature combined with our Python engine:

1. **Expert pre-loading**: Our Python dispatcher pre-fetches expert weights into a shared memory region or temp files before each layer's compute.

2. **Server/subprocess mode**: Drive `llama-server` or `llama-cli` via subprocess, using the `--override-tensor` or `--tensor-split` flags to route expert layers to CPU.

3. **Fallback — minimal C patch**: If the above is insufficient, write the smallest possible patch to `llama.cpp` source (document each change) and build via CMake only in that case.

### 5.4 Key interactions with llama.cpp binary

```
vendor/bin/llama-cli.exe    — run inference via subprocess
vendor/bin/llama-bench.exe  — baseline benchmarks
vendor/bin/llama-server.exe — optional HTTP API mode
```

### 5.5 Step-by-step

#### Step 1: Obtain llama.cpp binaries

- [ ] Write `setup.py`: fetch latest Vulkan release zip from `https://github.com/ggml-org/llama.cpp/releases`, verify SHA256, extract to `vendor/bin/`
- [ ] Add `vendor/llama.cpp` as a git submodule pinned to the same tag as the downloaded binaries
- [ ] Verify baseline: `vendor/bin/llama-bench.exe -m Qwen3-30B-A3B-Q4_K_M.gguf -ngl 99 -p 512 -n 128`
- [ ] Record baseline tok/s (pp and tg)

#### Step 2: Python inference driver

- [ ] Write `inference_driver.py`: wraps `llama-cli.exe` subprocess, parses stdout token stream
- [ ] Accept `--experts-dir`, `--cache-gb`, `--model-base` arguments
- [ ] Before launching subprocess, call `SetProcessWorkingSetSize` via ctypes

#### Step 3: Expert weight injection

- [ ] Investigate whether `llama-cli.exe` `--no-mmap --mlock` + expert CPU offload flags suffice to load `model_base.gguf` without expert data
- [ ] If yes: Python dispatcher pre-stages expert files; llama.cpp reads them normally
- [ ] If no: write minimal C patch, build from submodule source, document why binary was insufficient

#### Step 4: Memory management

- [ ] At engine startup (Python): call `SetProcessWorkingSetSize(20GB, 26GB)` via ctypes on the llama.cpp subprocess handle
- [ ] Monitor committed memory via `psutil` during inference

**Test: `test_integration_llamacpp.py`** (end-to-end, uses subprocess)
```
- [ ] setup.py: vendor/bin/llama-cli.exe exists and prints version without error
- [ ] Smoke test: generate 10 tokens with streaming engine.
      Output is coherent English (not garbage).
- [ ] Determinism: generate same prompt twice with same seed.
      Output matches.
- [ ] Correctness: generate 50 tokens with streaming engine.
      Compare against baseline (full GGUF) output.
      Outputs must be identical (same quantized weights, same compute).
      If not identical, investigate: may be floating point ordering differences.
      Relaxed check: first 10 tokens must match.
- [ ] Memory: during inference, committed memory stays under 28 GB (psutil).
- [ ] Disk writes: monitor SSD write counter. Zero writes from our process
      during entire inference run.
- [ ] Cold start: first token latency with empty cache.
- [ ] Warm steady state: tok/s after 50 tokens (cache populated).
- [ ] Expert offload config: run with --no-mmap --mlock and
      experts loaded via streaming. Verify all attention on GPU (Vulkan),
      all expert compute on CPU.
```

### Integration test for Phase 2C

```
- [ ] Long generation (500 tokens): no crashes, no memory growth, 
      coherent output throughout.
- [ ] Chat mode: multi-turn conversation (3 turns), 
      verify context is maintained correctly.
- [ ] Compare tok/s: streaming engine vs baseline (full GGUF in RAM).
      Expected: streaming with warm cache ≈ baseline (within 10%).
      Cold cache ≈ 50-80% of baseline (I/O overhead).
```

---

## 6. Phase 2D — Benchmarking Suite

### 6.1 Objective

Systematic measurement of performance characteristics to validate the technique and establish scaling predictions.

### 6.2 Benchmark configurations

```
Config A: Baseline — Full GGUF in RAM, standard llama.cpp, Vulkan
Config B: Streaming, unlimited cache — All experts fit in cache (~13 GB)
Config C: Streaming, 4 GB cache — Forces ~70% cache hit rate
Config D: Streaming, 1 GB cache — Forces ~8% cache hit rate (simulates large model)
Config E: Streaming, 256 MB cache — Near-zero cache (pure SSD streaming)
```

### 6.3 Metrics to collect

For each configuration, run: `prompt="Explain quantum computing in detail" tokens=200`

```
- tok/s (text generation, after prompt processing)
- Time to first token (TTFT)
- Cache hit rate (overall and per-layer)
- Average I/O latency per layer (ms)
- Average compute latency per layer (ms)
- Peak committed memory (MB)
- Total bytes read from SSD
- Total bytes written to SSD (must be zero)
- CPU utilization (%)
- GPU utilization (%) — via Vulkan timestamps if available
```

### 6.4 Step-by-step

- [ ] Implement `benchmark_runner.py` that runs each config and collects metrics
- [ ] Implement `expert_cache_stats_dump()` that writes per-layer hit rates to JSON
- [ ] Run each config 3 times, report mean and stddev
- [ ] Generate comparison table and charts

**Test: `test_benchmark_suite.py`**
```
- [ ] Config A runs successfully and produces valid metrics
- [ ] Config B tok/s is within 10% of Config A (cache holds everything)
- [ ] Config C → D → E shows monotonic decrease in tok/s
- [ ] Config E tok/s is bounded by SSD throughput: 
      expected_tps ≈ ssd_bandwidth / expert_bytes_per_token
- [ ] All configs: disk writes == 0
- [ ] All configs: committed memory ≤ cache_limit + base_overhead + 2 GB tolerance
- [ ] Cache hit rate for Config B ≈ 100% (after warmup)
- [ ] Cache hit rate for Config E < 10%
```

---

## 7. Phase 3 — Optimizations

Execute only after Phase 2 is complete and validated.

### 7.1 Prefetch speculation

While layer N's experts are computing, predict and pre-load layer N+1's experts.

- Heuristic 1: Load the same expert IDs (experts often repeat across adjacent layers)
- Heuristic 2: Load the globally most popular experts for layer N+1
- Measure: prefetch accuracy (% of prefetched experts actually used)

### 7.2 Expert file consolidation

Instead of 6,144 individual files, group by layer: `blk00_experts.bin`, `blk01_experts.bin`, ...

Each file contains all 128 experts for that layer, contiguously. On dispatch:
- Open one file per layer (keep handle open)
- Seek to expert offset within the file
- Issue `ReadFile` at the correct offset

Benefits: fewer file opens, larger sequential reads, better NVMe queue depth.

### 7.3 ML-based cache replacement

The FlashMoE paper (arXiv 2601.17063, Jan 2026) showed that an ML-based cache policy improves hit rate by up to 51% over LRU. Implement their lightweight predictor:

- Input: recent expert activation history (last N tokens)
- Output: probability each expert will be activated in the next token
- Evict the expert with lowest predicted future use

### 7.4 Vulkan staging buffer zero-copy

If Vulkan exposes host-visible device-local memory (which it does on UMA architectures like Lunar Lake), read SSD data directly into a Vulkan buffer that both CPU and GPU can access.

- Use `vkAllocateMemory` with `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`
- Map the buffer, read SSD data into it, GPU reads directly
- No CPU→GPU copy needed

### 7.5 NPU offload for routing

The Lunar Lake NPU 4 delivers 48 TOPS (INT8). Routing is a small linear projection (2048 → 128) followed by softmax and top-K. This could run on the NPU while GPU handles attention and CPU handles expert compute.

Requires: OpenVINO IR model for the router, async dispatch.

---

## 8. Scaling Path to DeepSeek V3.2

### 8.1 Architecture comparison

| Property | Qwen3-30B-A3B (PoC) | DeepSeek V3.2 (target) | Factor |
|---|---|---|---|
| Total params | 30.5B | 685B | 22x |
| Active params/token | 3.3B | 37B | 11x |
| Experts per layer | 128 | 256 + 1 shared | 2x |
| K (active experts) | 8 | 8 | 1x |
| Expert size (Q4) | ~500 KB | ~22 MB | 44x |
| Expert size (Q2) | ~250 KB | ~11 MB | 44x |
| I/O per token | ~192 MB | ~5.1 GB (Q2) | 27x |
| Resident weights | ~6 GB | ~8-17 GB | 2-3x |
| Attention | GQA | MLA + DSA | Complex |
| Expert total volume | ~13 GB | ~163 GB (Q2) | 13x |

### 8.2 What transfers directly

- Expert extraction pipeline (adapt tensor names and shapes)
- I/O pool (same `ReadFile` + `FILE_FLAG_NO_BUFFERING` pattern)
- Cache LRU (same algorithm, just larger files and fewer fit)
- Benchmark suite (same metrics, different expected values)

### 8.3 What needs new work

- **MLA attention implementation**: DeepSeek V3.2 uses Multi-head Latent Attention with KV compression. Not in standard llama.cpp.
- **DeepSeek Sparse Attention (DSA)**: Fine-grained sparse attention for long contexts. V3.2-specific.
- **Shared expert handling**: DeepSeek has 1 always-active shared expert per layer. Must be kept in RAM.
- **Much larger I/O**: 5.1 GB/token at Q2 with 6 GB/s SSD = ~850 ms/token baseline. Pipeline overlap becomes essential, not optional.
- **Cache policy critical**: With only ~16% of experts fitting in 20 GB cache, hit rate determines viability. ML-based caching from Phase 3 becomes mandatory.

### 8.4 Expected performance on Lunar Lake

```
DeepSeek V3.2, Q2_K, cache hit rate 80%:
  I/O per token: 5.1 GB × (1 - 0.80) = 1.02 GB from SSD
  I/O time: 1.02 GB / 6 GB/s ≈ 170 ms
  Compute time: ~200-400 ms (CPU expert matmul + GPU attention)
  Total: ~370-570 ms/token
  Throughput: ~1.7-2.7 tok/s

DeepSeek V3.2, Q2_K, cache hit rate 50%:
  I/O per token: 5.1 GB × 0.50 = 2.55 GB from SSD
  I/O time: 2.55 GB / 6 GB/s ≈ 425 ms
  Total: ~625-825 ms/token
  Throughput: ~1.2-1.6 tok/s
```

Usable for batch/offline processing. Not interactive, but functional.

---

## 9. Reference Materials

### Papers
1. **LLM in a Flash** (Apple, 2023): arXiv 2312.11514 — Theoretical foundation for SSD-based LLM inference
2. **FlashMoE** (Jan 2026): arXiv 2601.17063 — SSD offloading with ML-based cache, tested Qwen3-30B-A3B
3. **KTransformers** (SOSP'25): CPU-GPU hybrid MoE inference with expert deferral
4. **HybriMoE** (Apr 2025): arXiv 2504.05897 — Dynamic scheduling over KTransformers

### Repositories
5. **flash-moe**: https://github.com/danveloper/flash-moe — Original Apple Silicon implementation
6. **llama.cpp**: https://github.com/ggml-org/llama.cpp — Inference engine (Vulkan backend)
7. **llama.cpp MoE guide**: https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide

### Windows API
8. **FILE_FLAG_NO_BUFFERING**: https://learn.microsoft.com/en-us/windows/win32/fileio/file-buffering
9. **SetProcessWorkingSetSize**: https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-setprocessworkingsetsize
10. **VirtualLock limitations** (Raymond Chen): https://devblogs.microsoft.com/oldnewthing/20071106-00/?p=24573

### Model
11. **Qwen3-30B-A3B**: https://huggingface.co/Qwen/Qwen3-30B-A3B
12. **GGUF quants**: https://huggingface.co/bartowski/Qwen_Qwen3-30B-A3B-GGUF
