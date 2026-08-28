# YMQ-Compiler (v2.0)

An architecture-aware post-training quantization compiler for Large Language Models running on the `llama.cpp` / GGUF framework. 

Standard quantization pipelines treat transformer weights like a flat, uniform dataset, applying destructive blanket low-bit compression to delicate internal routing mechanisms. The **YMQ-Compiler** solves high-context formatting loops, text degradation, and "finish-line amnesia" by parsing model matrices dynamically using a multi-tiered data-science evaluation matrix.

Developed as an in-house AI optimization utility for **ZeroDigest** indie game studio and local developer workspaces.

---

## 🛠️ Deep Engineering Core Mechanics

The compiler bypasses traditional percentage-based truncation loops by applying advanced, layer-by-layer parameter isolation:

### 1. Log-Space Gap Detection Clustering
Instead of looking at raw linear weights where massive 70k+ cognitive spikes drown out mid-tier reasoning signals, the engine calculates statistical cluster variances in log-space (measuring orders of magnitude). It dynamically isolates intermediate logic spikes (scoring 72 to 512) and elevates them to high-fidelity non-linear 4-bit (`IQ4_XS`) formats, while safely collapsing idle factual background layers down to aggressive 2-bit floors (`IQ2_S`) to maximize bit economy.

### 2. Fading Boundary Tapering
Recognizing the extreme structural fragility of initial token entry data vectors, the compiler forces an input wave cushion (`L00=IQ4_NL` -> `L01=IQ4_XS` -> `L02=IQ3_XXS`) that gradually stabilizes parameters before hitting the fallback pools. The final exit gate layer (`max_layer`) is locked strictly to `IQ4_NL` to guarantee an error-free handshake to the vocabulary head.

### 3. Dedicated Parallel Gate Insulation
The script automatically screens the size and naming footprints of delicate parallel Attention and Mamba Linear State Space Model (SSM) routing networks. Tiny tracking engines (`attn_k`, `attn_v`, `ssm_beta`) are kept at full unquantized native resolution (`Q8_0`/`COPY`), completely preventing cumulative context drift during deep prompts.

### 4. Asymmetric Vocabulary Protection
Forces both tied-weight embedding arrays (`token_embd.weight` and `output.weight`) into pristine, matching high-bit resolution matrices. This gives your agent's prediction head the necessary precision to read closing brackets, format strict system command keys, and trigger special Jinja template stop-tokens cleanly past 43kb+ contexts.

---

## 📊 Standard Preset Scale Blueprints

The compiler features 5 distinct automated clothing-size presets tailored for diverse VRAM budgets:

*   **`XXS`**: Max bit economy. Core blocks use aggressive `IQ2_XXS` baselines. Best suited for tight single-GPU hardware configurations.
*   **`XS`**: Balanced economy. Features `IQ4_NL` on primary peaks and `IQ3_S` on intermediate ridges.
*   **`M`**: **The Ultimate Coding Sweet Spot.** Elevates background fallbacks to `IQ3_XXS` and primary logic spikes to `Q5_K`. Fully immune to high-context amnesia loops.
*   **`L`**: Premium single-GPU processing. Heavy 6-bit shields and full 4-bit non-linear fallbacks.
*   **`XL`**: Maximum fidelity. Complete structural matrix protection for massive multi-file codebase parsing tasks.

---

## 📦 Installation

### Prerequisites

| Dependency | Purpose | Required |
|------------|---------|----------|
| `python3` (≥ 3.8) | Runs the embedded analysis engine (GGUF header parsing, imatrix scoring) | ✅ Yes |
| `gguf-py` | Python library for reading GGUF tensor metadata (`from gguf.gguf_reader import GGUFReader`) | ✅ Yes |
| `llama-quantize` | The llama.cpp quantization binary that performs the actual per-tensor quantization | ✅ Yes |
| `strings` (binutils) | Architecture auto-detection (MoE vs dense) from imatrix tensor names | ⚠️ Optional — defaults to `dense` if missing |

### 1. Install Python dependencies

```bash
pip3 install gguf
```

### 2. Obtain the compiler script

Clone or copy `ymq-compile.sh` into your working directory:

```bash
git clone <repo-url> ymq-compiler   # or download ymq-compile.sh directly
cd ymq-compiler
chmod +x ymq-compile.sh
```

### 3. Prepare your input files

The compiler expects **two** input files:

| File | Description |
|------|-------------|
| `model.imatrix.gguf` | An importance-matrix file generated with `llama-bench -m <model> -im <corpus>` (or your preferred imatrix tool). Must contain per-layer tensor importance scores. |
| `model.gguf` | The source model in **16-bit** format (`F16`, `BF16`, or `Q8_0`). This is the quantization baseline — the compiler reads its tensor layout and writes the final output alongside it. Higher-precision sources (F16/BF16) yield better results than Q8_0 since they preserve more weight information before per-tensor quantization. |

> **Note:** The script auto-detects whether the model is **Dense** or **MoE/Hybrid** (by inspecting `*_exps` tensor names in the imatrix) and applies the matching preset table automatically. No manual architecture flag needed.

### 4. Place `llama-quantize` on your PATH

The generated `run_quant.sh` invokes `./llama-quantize`, so the binary must be **in the same directory** from which you execute the script (or available via a relative path). A typical setup:

```bash
# Build llama.cpp once (if not already built)
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build && cmake --build build --config Release -j
export PATH="$(pwd)/build/bin:$PATH"   # or copy llama-quantize next to ymq-compile.sh
```

### 5. Verify the installation

Run a quick smoke test — the script will print its detected architecture, preset targets, and an estimated output size before writing anything:

```bash
./ymq-compile.sh \
    "/path/to/model.imatrix.gguf" \
    "/path/to/model-bf16.gguf" \
    --preset m 2>&1 | head -20
```

If you see a line like:

```
YMQ-Compiler: Detected architecture: moe | Preset [M] targets - INPUT=Q3_K, HIGH=Q5_K, ...
```

the installation is working correctly.

---

## 🚀 Execution & Command-Line Usage

The automation script is a drop-in compiler wrapper tool. Pass your `.imatrix` file data log and target preset straight through the launcher switch flags:

```bash
./ymq-compile.sh \
    "/path/to/model.imatrix.gguf" \
    "/path/to/model.gguf" \
    --preset m
```

The script **does not** run quantization directly. Instead it:

1. Parses the GGUF tensor layout and imatrix importance profiles.
2. Assigns per-tensor quantization targets (high/mid/low/floor tiers).
3. Writes a ready-to-run `run_quant.sh` into a model-named subdirectory (`<MODEL_NAME>/run_quant.sh`).

To actually produce the quantized GGUF, execute the generated script from the directory containing your `llama-quantize` binary:

```bash
cd <MODEL_NAME>/
./run_quant.sh
```

The final output is written as `<model>-YMQ-<PRESET>.gguf` in the same directory as the source 16-bit file.

---

## 🤗 Quantized Models

Pre-compiled, architecture-aware clothing-size variants (XXS, XS, M, L, XL) featuring native Multi-Token Prediction (MTP) support are hosted officially on Hugging Face.

👉 **[Download Presets from the ZeroDigest Hub](https://huggingface.co/zerodigest)**

---

## 📜 License

This project framework is open-source software licensed under the **MIT License**. It features an absolute liability shield providing software "AS IS", allowing for permissionless distribution, modification, and commercial application while preserving author attribution notices.

See the `LICENSE` file template for full legal terms.
