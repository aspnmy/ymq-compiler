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

## 🚀 Execution & Command-Line Usage

The automation script is a drop-in compiler wrapper tool. Pass your `.imatrix` file data log and target preset straight through the launcher switch flags:

```bash
chmod +x ymq-compile.sh
./ymq-compile.sh \
    "/path/to/model.imatrix.gguf" \
    "/path/to/raw_model_bf16.gguf" \
    --preset m
```

---

## 📜 License

This project framework is open-source software licensed under the **MIT License**. It features an absolute liability shield providing software "AS IS", allowing for permissionless distribution, modification, and commercial application while preserving author attribution notices.

See the `LICENSE` file template for full legal terms.
