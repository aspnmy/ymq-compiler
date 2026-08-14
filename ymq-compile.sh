#!/usr/bin/env bash
# ==============================================================================
# YMQ-Compiler: Adaptive Imatrix-Driven GGUF Quantization Compiler
# Dynamically assigns per-layer quantization targets based on importance matrix
# Supports configurable size thresholds, score-based tiering, and full type coverage
# ==============================================================================
set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <path_to_imatrix.gguf> <path_to_source_q8_0.gguf> [--preset=xxs|xs|m|l|xl] [--input-target=Q4_K_M] [--high-target=Q5_K] [--mid-target=IQ4_XS] [--low-target=IQ3_S] [--copy-target=COPY] [--tiny-target=COPY] [--default-target=IQ2_XXS] [--small-threshold=1.0] [--tiny-threshold=0.1]"
    exit 1
fi

IMATRIX_PATH="$1"
SOURCE_GGUF="$2"
shift 2
# Output GGUF saved alongside input Q8 file
OUTPUT_GGUF_DIR="$(dirname "$SOURCE_GGUF")"
# run_quant.sh saved in a model-named folder alongside this script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_NAME="$(basename "${SOURCE_GGUF%.gguf}")"
OUTPUT_DIR="${SCRIPT_DIR}/${MODEL_NAME}"
SMALL_THRESHOLD="1.0"
TINY_THRESHOLD="0.1"

# Auto-detect model architecture from imatrix tensor names.
# MoE and Hybrid models share the same preset set (both have 'exps' tensors).
if strings "$IMATRIX_PATH" 2>/dev/null | grep -qi "ffn_down_exps\|ffn_gate_exps\|ffn_up_exps"; then
    MODEL_ARCH="moe"   # Covers pure MoE and Hybrid (MoE + dense FFN) models
else
    MODEL_ARCH="dense"
fi

# Preset Set A: Dense models
apply_dense_preset() {
    case "$1" in
        xxs) PRESET_NAME="XXS"; INPUT_TARGET="Q3_K"; COPY_TARGET="Q5_K"; TINY_TARGET="Q8_0"; HIGH_TARGET="IQ4_XS"; MID_TARGET="IQ3_XXS"; LOW_TARGET="IQ2_S"; DEFAULT_TARGET="IQ2_XXS" ;;
        xs)  PRESET_NAME="XS";  INPUT_TARGET="Q3_K"; COPY_TARGET="Q6_K"; TINY_TARGET="Q8_0"; HIGH_TARGET="IQ4_NL"; MID_TARGET="IQ3_S"; LOW_TARGET="IQ3_XXS"; DEFAULT_TARGET="IQ2_S" ;;
        m)   PRESET_NAME="M";   INPUT_TARGET="Q3_K"; COPY_TARGET="Q6_K"; TINY_TARGET="Q8_0"; HIGH_TARGET="Q5_K";   MID_TARGET="IQ4_XS"; LOW_TARGET="IQ3_S";  DEFAULT_TARGET="IQ3_XXS" ;;
        l)   PRESET_NAME="L";   INPUT_TARGET="Q4_K"; COPY_TARGET="Q6_K"; TINY_TARGET="Q8_0"; HIGH_TARGET="Q6_K";   MID_TARGET="Q5_K";   LOW_TARGET="IQ4_NL"; DEFAULT_TARGET="IQ3_S" ;;
        xl)  PRESET_NAME="XL";  INPUT_TARGET="Q4_K"; COPY_TARGET="Q6_K"; TINY_TARGET="Q8_0"; HIGH_TARGET="Q6_K";   MID_TARGET="Q6_K";   LOW_TARGET="Q5_K";   DEFAULT_TARGET="IQ4_NL" ;;
        *) echo "ERROR: Unknown preset '$1'. Available presets: xxs, xs, m, l, xl" >&2; exit 1 ;;
    esac
}

# Preset Set B: MoE / Hybrid models (exps-based architectures)
apply_moe_preset() {
    case "$1" in
        xxs) PRESET_NAME="XXS"; INPUT_TARGET="Q3_K"; COPY_TARGET="Q4_K"; TINY_TARGET="Q8_0"; HIGH_TARGET="IQ3_XXS"; MID_TARGET="IQ2_S"; LOW_TARGET="IQ2_XXS"; DEFAULT_TARGET="IQ2_XXS" ;;
        xs)  PRESET_NAME="XS";  INPUT_TARGET="Q3_K"; COPY_TARGET="Q5_K"; TINY_TARGET="Q8_0"; HIGH_TARGET="IQ4_XS";   MID_TARGET="IQ3_XS"; LOW_TARGET="IQ2_S";  DEFAULT_TARGET="IQ2_XXS" ;;
        m)   PRESET_NAME="M";   INPUT_TARGET="Q3_K"; COPY_TARGET="Q6_K"; TINY_TARGET="Q8_0"; HIGH_TARGET="IQ4_NL";   MID_TARGET="IQ3_S"; LOW_TARGET="IQ2_S";  DEFAULT_TARGET="IQ2_XS" ;;
        l)   PRESET_NAME="L";   INPUT_TARGET="Q4_K"; COPY_TARGET="Q6_K"; TINY_TARGET="Q8_0"; HIGH_TARGET="Q5_K";   MID_TARGET="IQ4_XS";   LOW_TARGET="IQ3_XXS"; DEFAULT_TARGET="IQ2_S" ;;
        xl)  PRESET_NAME="XL";  INPUT_TARGET="Q4_K"; COPY_TARGET="Q6_K"; TINY_TARGET="Q8_0"; HIGH_TARGET="Q6_K";   MID_TARGET="IQ4_NL";   LOW_TARGET="IQ3_XS";   DEFAULT_TARGET="IQ3_XXS" ;;
        *) echo "ERROR: Unknown preset '$1'. Available presets: xxs, xs, m, l, xl" >&2; exit 1 ;;
    esac
}

# Apply preset based on detected architecture (single if block)
apply_preset() {
    if [ "$MODEL_ARCH" = "moe" ]; then
        apply_moe_preset "$1"
    else
        apply_dense_preset "$1"
    fi
}

# Default to 'm' (medium) preset - auto-selects dense or moe targets based on detected architecture
apply_preset "m"

# Parse optional arguments (support both --arg=val and --arg val)
while [ "$#" -gt 0 ]; do
    case $1 in
        --preset) apply_preset "$2"; shift 2 ;;
        --preset=*) apply_preset "${1#*=}"; shift ;;
        --high-target) HIGH_TARGET="$2"; shift 2 ;;
        --high-target=*) HIGH_TARGET="${1#*=}"; shift ;;
        --mid-target) MID_TARGET="$2"; shift 2 ;;
        --mid-target=*) MID_TARGET="${1#*=}"; shift ;;
        --low-target) LOW_TARGET="$2"; shift 2 ;;
        --low-target=*) LOW_TARGET="${1#*=}"; shift ;;
        --copy-target) COPY_TARGET="$2"; shift 2 ;;
        --copy-target=*) COPY_TARGET="${1#*=}"; shift ;;
        --tiny-target) TINY_TARGET="$2"; shift 2 ;;
        --tiny-target=*) TINY_TARGET="${1#*=}"; shift ;;
        --default-target) DEFAULT_TARGET="$2"; shift 2 ;;
        --default-target=*) DEFAULT_TARGET="${1#*=}"; shift ;;
        --small-threshold) SMALL_THRESHOLD="$2"; shift 2 ;;
        --small-threshold=*) SMALL_THRESHOLD="${1#*=}"; shift ;;
        --tiny-threshold) TINY_THRESHOLD="$2"; shift 2 ;;
        --tiny-threshold=*) TINY_THRESHOLD="${1#*=}"; shift ;;
        --input-target) INPUT_TARGET="$2"; shift 2 ;;
        --input-target=*) INPUT_TARGET="${1#*=}"; shift ;;
        *) echo "WARNING: Unknown argument '$1' ignored" >&2; shift ;;
    esac
done

echo "YMQ-Compiler: Detected architecture: $MODEL_ARCH | Preset [$PRESET_NAME] targets - INPUT=$INPUT_TARGET, HIGH=$HIGH_TARGET, MID=$MID_TARGET, LOW=$LOW_TARGET, COPY=$COPY_TARGET, TINY=$TINY_TARGET, DEFAULT=$DEFAULT_TARGET"

# Strip .Q8_0 suffix from model name for cleaner output (e.g. model.Q8_0.gguf -> model-YMQ-XXS.gguf)
MODEL_BASENAME="$(basename "${SOURCE_GGUF%.gguf}")"
MODEL_BASENAME="${MODEL_BASENAME%.Q8_0}"
OUTPUT_GGUF="${OUTPUT_GGUF_DIR}/${MODEL_BASENAME}-YMQ-${PRESET_NAME}.gguf"

echo "YMQ-Compiler: Running adaptive imatrix scan and precision estimation..."

# We capture the standard output containing the multi-line tensor types directly
QUANT_ARGS=$(python3 - "$INPUT_TARGET" "$HIGH_TARGET" "$MID_TARGET" "$LOW_TARGET" "$COPY_TARGET" "$TINY_TARGET" "$DEFAULT_TARGET" "$SMALL_THRESHOLD" "$TINY_THRESHOLD" <<EOF
import re
import os
import struct
import sys
from collections import defaultdict

# ==============================================================================
# Configuration and Constants
# ==============================================================================

# YMQ BPW Map: Bits-per-weight for all supported quantization types
BPW_MAP = {
    # Full quantization type BPW mapping
    "F32": 32.0, "F16": 16.0, "BF16": 16.0,
    "Q8_0": 8.5,
    "Q6_K": 6.5, "Q5_K": 5.5, "Q5_K_S": 5.5, "Q5_K_M": 5.5,
    "Q5_0": 5.0, "Q5_1": 5.5,
    "Q4_K_M": 4.5, "Q4_K": 4.5, "Q4_K_S": 4.5,
    "Q4_0": 4.0, "Q4_1": 4.5,
    "Q3_K_M": 3.8, "Q3_K": 3.8, "Q3_K_S": 3.5, "Q3_K_L": 3.8,
    "Q2_K": 3.0, "Q2_K_S": 3.0,
    "Q2_0": 2.25, "Q1_0": 1.125,
    "IQ4_NL": 4.5, "IQ4_XS": 4.25,
    "IQ3_XXS": 3.06, "IQ3_XS": 3.3, "IQ3_S": 3.44, "IQ3_M": 3.66,
    "IQ2_XXS": 2.06, "IQ2_XS": 2.31, "IQ2_S": 2.5, "IQ2_M": 2.7,
    "IQ1_S": 1.56, "IQ1_M": 1.75,
    "TQ1_0": 1.69, "TQ2_0": 2.06,
    "MXFP4_MOE": 4.0,
    "COPY": 8.5  # COPY preserves original Q8 data, same size as Q8_0
}

# YMQ Tensor Pattern Normalizer: Maps variant tensor names to canonical base names
PATTERN_TO_BASE = {
    'ffn_down_exps': 'ffn_down',
    'ffn_down': 'ffn_down',
    'ssm_out': 'ssm_out',
    'ssm_in': 'ssm_in',
    'attn_output': 'attn_output',
    'attn_q': 'attn_q',
    'attn_k': 'attn_k',
    'attn_v': 'attn_v',
    'attn_qkv': 'attn_qkv',
    'attn_gate': 'attn_gate',
    'ssm_a': 'ssm_a',
    'ssm_alpha': 'ssm_alpha',
    'ssm_beta': 'ssm_beta',
    'ssm_conv1d': 'ssm_conv1d',
    'ssm_dt': 'ssm_dt',
    'ffn_gate_inp': 'ffn_gate_inp',
}


# ==============================================================================
# YMQ Stage 1: Model Analysis & Imatrix Extraction
# ==============================================================================
def ymq_stage1_analysis(imatrix_path, gguf_path):
    """Parse GGUF tensor layout, extract imatrix importance profiles per layer."""
    
    if not os.path.exists(imatrix_path) or not os.path.exists(gguf_path):
        print("ERROR: Input file paths missing", file=sys.stderr)
        sys.exit(1)
    
    # --- Parse GGUF header and extract tensor sizes ---
    tensor_element_counts = {}
    
    print("DEBUG: Starting GGUF header parsing with GGUFReader...", file=sys.stderr)
    
    try:
        # Add gguf-py to path if available, otherwise use system installation
        # Note: __file__ is not defined when code is piped via stdin (python3 -), so we skip local path lookup
        
        from gguf.gguf_reader import GGUFReader
        
        print(f"DEBUG: Reading GGUF file: {gguf_path}", file=sys.stderr)
        reader = GGUFReader(gguf_path)
        
        # Extract tensor information from reader
        for tensor in reader.tensors:
            tensor_name = tensor.name
            n_elements = tensor.n_elements
            
            # Get tensor type - handle both gguf-py versions
            try:
                tensor_type = tensor.tensor_type.name
            except AttributeError:
                tensor_type = str(tensor.tensor_type)
            
            # Sanity check - reasonable tensor sizes
            if 0 < n_elements < 1000000000000:
                tensor_element_counts[tensor_name] = n_elements
        
        print(f"DEBUG: Found {len(tensor_element_counts)} tensors in GGUF file", file=sys.stderr)
        
    except ImportError as e:
        print(f"ERROR: Failed to import gguf-py library: {e}", file=sys.stderr)
        print("Please install it with: pip3 install gguf", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"DEBUG: Exception during tensor parsing: {e}", file=sys.stderr)
        print(f"DEBUG: Traceback: {traceback.format_exc()}", file=sys.stderr)
    
    print(f"DEBUG: Total tensors parsed: {len(tensor_element_counts)}", file=sys.stderr)
    
    # --- Extract layer-specific tensor sizes from parsed data ---
    layer_dimensions = {}
    
    for tensor_name, elem_count in tensor_element_counts.items():
        # Match layer tensors (blk.N.name.weight or blk.N.name)
        match = re.match(r'blk\.(\d+)\.(.+)', tensor_name)
        if match:
            layer_num = int(match.group(1))
            base_tensor = match.group(2).lower()
            
            # Normalize tensor name for lookup
            for pattern, base in PATTERN_TO_BASE.items():
                if base_tensor.startswith(pattern):
                    layer_dimensions[f"{layer_num}_{base}"] = elem_count
                    break
    
    # --- Read Imatrix metadata profiles ---
    with open(imatrix_path, 'rb') as f:
        imatrix_data = f.read(1024 * 1024 * 16)
    
    pattern_im = rb'blk\.(\d+)\.(ffn_down_exps|ffn_down|ssm_out|ssm_in)\.weight'
    matches = list(re.finditer(pattern_im, imatrix_data, re.IGNORECASE))
    
    layer_ffn_metrics = {}
    layer_ssm_metrics = {}
    # layer_types stores a SET of types per key to support hybrid layers (both MoE + dense)
    layer_types = defaultdict(set)
    
    for match in matches:
        layer_num = int(match.group(1).decode('utf-8'))
        tensor_name = match.group(2).decode('utf-8').lower()
        start_offset = match.end()
        
        chunk_data = imatrix_data[start_offset:start_offset + 1024]
        floats = []
        for offset in range(0, len(chunk_data) - 4, 4):
            val_tuple = struct.unpack('<f', chunk_data[offset:offset+4])
            val = val_tuple[0]
            if 0.00001 < abs(val) < 100000.0:
                floats.append(abs(val))
                
        if floats:
            current_mean = sum(floats) / len(floats)
            if "ssm" in tensor_name:
                layer_ssm_metrics[layer_num] = max(layer_ssm_metrics.get(layer_num, 0), current_mean)
                layer_types[f"{layer_num}_ssm"].add("mamba")
            else:
                layer_ffn_metrics[layer_num] = max(layer_ffn_metrics.get(layer_num, 0), current_mean)
                new_type = "_exps" if "exps" in tensor_name else "dense"
                layer_types[f"{layer_num}_ffn"].add(new_type)
    
    # --- Calculate aggregate metrics ---
    all_ffn_vals = list(layer_ffn_metrics.values()) if layer_ffn_metrics else []
    all_ssm_vals = list(layer_ssm_metrics.values()) if layer_ssm_metrics else []
    
    all_moe_vals = [val for l, val in layer_ffn_metrics.items() if "_exps" in layer_types.get(f"{l}_ffn", set())]
    all_dense_vals = [val for l, val in layer_ffn_metrics.items() if "dense" in layer_types.get(f"{l}_ffn", set())]
    
    # Calculate max_layer from actual parsed tensor data
    if layer_ffn_metrics or layer_ssm_metrics:
        max_layer = max(list(layer_ffn_metrics.keys()) + list(layer_ssm_metrics.keys()))
    else:
        # Fallback: derive from actual tensor_element_counts if imatrix parsing failed
        layer_nums = []
        for tensor_name in tensor_element_counts.keys():
            match = re.match(r'blk\.(\d+)\.', tensor_name)
            if match:
                layer_nums.append(int(match.group(1)))
        max_layer = max(layer_nums) if layer_nums else 0
    
    # Additional fallback - use actual parsed data only
    if max_layer == 0 and len(tensor_element_counts) > 0:
        for tensor_name in tensor_element_counts.keys():
            match = re.match(r'blk\.(\d+)\.', tensor_name)
            if match:
                layer_num = int(match.group(1))
                if layer_num > max_layer:
                    max_layer = layer_num
    
    peak_moe = max(all_moe_vals) if all_moe_vals else None
    peak_dense = max(all_dense_vals) if all_dense_vals else None
    peak_ssm = max(all_ssm_vals) if all_ssm_vals else None
    avg_ffn = sum(all_ffn_vals) / len(all_ffn_vals) if all_ffn_vals else 0
    
    source_bytes = os.path.getsize(gguf_path)
    has_ssm = len(layer_ssm_metrics) > 0
    
    return {
        'tensor_element_counts': tensor_element_counts,
        'layer_dimensions': layer_dimensions,
        'imatrix_data': imatrix_data,
        'gguf_path': gguf_path,
        'layer_ffn_metrics': layer_ffn_metrics,
        'layer_ssm_metrics': layer_ssm_metrics,
        'layer_types': layer_types,
        'max_layer': max_layer,
        'peak_moe': peak_moe,
        'peak_dense': peak_dense,
        'peak_ssm': peak_ssm,
        'avg_ffn': avg_ffn,
        'source_bytes': source_bytes,
        'has_ssm': has_ssm
    }


# ==============================================================================
# YMQ Stage 2: Adaptive Target Assignment & Script Generation
# ==============================================================================
def ymq_stage2_assign_targets(data, input_target, high_target, mid_target, low_target, copy_target, tiny_target, default_target, small_threshold_gb, tiny_threshold_gb):
    """YMQ Core Algorithm: Assign quantization targets dynamically.

    Architecture-Agnostic Logic (NO hardcoded array names):
    - Discover all unique tensor arrays from parsed GGUF data
    - Classify by Q8 cumulative size into 3 tiers (tiny/small/large)
    - Tiny arrays (< tiny_threshold GB) -> TINY_TARGET
    - Small arrays (tiny_threshold - small_threshold GB) -> COPY_TARGET
    - Large arrays (>= small_threshold GB) -> imatrix-scored per-layer targeting
      (FFN metrics for FFN tensors, SSM metrics for SSM tensors, default otherwise)
    """

    max_layer = data['max_layer']
    tensor_element_counts = data['tensor_element_counts']
    layer_dimensions = data['layer_dimensions']
    layer_ffn_metrics = data['layer_ffn_metrics']
    layer_ssm_metrics = data['layer_ssm_metrics']
    layer_types = data['layer_types']
    has_ssm = data['has_ssm']
    peak_moe = data['peak_moe']
    peak_dense = data['peak_dense']
    peak_ssm = data['peak_ssm']
    avg_ffn = data['avg_ffn']

    INPUT_TARGET = input_target     # For token_embd.weight (input embedding)
    HIGH_TARGET = high_target       # For highest-scoring layers (>= 70% of peak)
    MID_TARGET = mid_target         # For mid-scoring layers (>= 15% of peak)
    LOW_TARGET = low_target         # For low-scoring layers (>= 40% of avg)
    COPY_TARGET = copy_target       # For small arrays (tiny_threshold - small_threshold Q8 size)
    TINY_TARGET = tiny_target       # For tiny arrays (< tiny_threshold Q8 size)
    DEFAULT_TARGET = default_target  # Configurable global default fallback
    Q8_BYTES_PER_ELEMENT = 1.0
    SMALL_THRESHOLD_BYTES = float(small_threshold_gb) * 1024.0 * 1024.0 * 1024.0
    TINY_THRESHOLD_BYTES = float(tiny_threshold_gb) * 1024.0 * 1024.0 * 1024.0

    # --- Norm / F32 patterns (not quantized aggressively) ---
    f32_tensors = ['rope_freqs.weight', 'output_norm.weight']
    norm_patterns = [
        'attn_norm', 'attn_q_norm', 'attn_k_norm', 'ffn_norm',
        'post_attention_norm', 'post_ffw_norm', 'layer_output_scale',
        'ssm_norm', 'ssm_a', 'ssm_conv1d', 'ssm_dt.bias', 'ffn_gate_inp'
    ]

    # --- Protected arrays with fixed targets (excluded from main algorithm) ---
    protected_patterns = ['attn_output']

    def is_protected_tensor(base_name):
        for p in protected_patterns:
            if base_name.startswith(p):
                return True
        return False

    def is_norm_tensor(base_name):
        for p in norm_patterns:
            if base_name.startswith(p):
                return True
        return False

    # --- Step 1: Discover all unique array base names from tensor_element_counts ---
    array_layers = defaultdict(dict)  # base_name -> {layer_num: n_elements}

    for tensor_name, elem_count in tensor_element_counts.items():
        match = re.match(r'blk\.(\d+)\.(.+)', tensor_name)
        if match:
            layer_num = int(match.group(1))
            base_name = match.group(2)
            array_layers[base_name][layer_num] = elem_count

    # --- Step 2: Classify arrays by Q8 size (three tiers) ---
    tiny_arrays = []   # Q8 < tiny_threshold -> TINY_TARGET
    small_arrays = []  # Q8 tiny_threshold - small_threshold -> COPY_TARGET
    large_arrays = []  # Q8 >= small_threshold -> imatrix-based targeting

    for base_name, layers_dict in array_layers.items():
        total_elements = sum(layers_dict.values())
        q8_size_bytes = total_elements * Q8_BYTES_PER_ELEMENT
        if q8_size_bytes < TINY_THRESHOLD_BYTES:
            tiny_arrays.append(base_name)
        elif q8_size_bytes < SMALL_THRESHOLD_BYTES:
            small_arrays.append(base_name)
        else:
            large_arrays.append(base_name)

    # --- Step 2b: Tapered layers (0,1,2) and end layer (max_layer) targets ---
    # Applied per-layer during imatrix processing with priority after small/tiny
    TAPERED_TARGETS = {0: HIGH_TARGET, 1: MID_TARGET, 2: LOW_TARGET}
    END_LAYER = max_layer
    END_TARGET = HIGH_TARGET

    # --- Step 3: Build base command parts ---
    cmd_parts = []

    # Fixed non-layer tensors (token_embd uses INPUT_TARGET, output uses COPY_TARGET)
    cmd_parts.append(f"--tensor-type token_embd.weight={INPUT_TARGET}")
    cmd_parts.append(f"--tensor-type output.weight={COPY_TARGET}")
    for p in f32_tensors:
        cmd_parts.append(f"--tensor-type {p}=F32")

    # F32 for norm patterns (wildcard) - use correct suffix per tensor type
    # Most tensors end in .weight, but ssm_a has no suffix and ssm_dt.bias already includes .bias
    norm_suffix_map = {
        'ssm_a': '',           # No suffix (buffer, not weight)
        'ssm_dt.bias': '',     # Already includes .bias
    }
    for p in norm_patterns:
        suffix = norm_suffix_map.get(p, '.weight')
        cmd_parts.append(f"--tensor-type blk.*.{p}{suffix}=F32")

    # Fixed protected tensors (wildcard) - set before dynamic processing
    cmd_parts.append(f"--tensor-type blk.*.attn_output.weight={COPY_TARGET}")

    # --- Step 4: Tiny arrays (Q8 < 0.1GB) -> TINY_TARGET ---
    for base_name in sorted(tiny_arrays):
        if is_norm_tensor(base_name):
            continue
        if is_protected_tensor(base_name):
            continue
        cmd_parts.append(f"--tensor-type blk.*.{base_name}={TINY_TARGET}")

    # --- Step 5: Small arrays (Q8 0.1GB - 1GB) -> COPY_TARGET ---
    for base_name in sorted(small_arrays):
        if is_norm_tensor(base_name):
            continue
        if is_protected_tensor(base_name):
            continue
        cmd_parts.append(f"--tensor-type blk.*.{base_name}={COPY_TARGET}")

    # --- Step 5b: Tapered/End layer targets applied during imatrix processing below ---
    # (handled inline in Step 6c per-layer assignment)

    # --- Step 6: Large arrays (Q8 >= 1GB) -> imatrix-based per-layer targeting ---
    # Determine which score source to use per array (no hardcoded names)
    def get_score_source(base_name):
        """Return (metrics_dict, peak_value, avg_value) for a given array base name.
        Returns (None, None, 0.0) if this array type has no relevant imatrix data."""
        lower = base_name.lower()
        # SSM arrays -> use SSM metrics only if 'ssm' is in the name
        if 'ssm' in lower:
            return layer_ssm_metrics, peak_ssm, 0.0
        # FFN-related arrays (ffn_down, ffn_gate, ffn_up, ffn_down_exps, etc.)
        if lower.startswith('ffn_'):
            if 'exps' in lower:
                return layer_ffn_metrics, peak_moe, avg_ffn
            else:
                return layer_ffn_metrics, peak_dense, avg_ffn
        # All other arrays have no imatrix data
        return None, None, 0.0

    target_rank = {INPUT_TARGET: 4.5, HIGH_TARGET: 5, COPY_TARGET: 6, TINY_TARGET: 7, MID_TARGET: 4, LOW_TARGET: 3, DEFAULT_TARGET: 2}

    def get_best_target(current, new_target):
        current_rank = target_rank.get(current, 0)
        new_rank = target_rank.get(new_target, 0)
        return new_target if new_rank > current_rank else current

    # --- Step 6a: Collect all layer scores across scored arrays ---
    all_layer_scores = {}
    array_score_info = {}

    for base_name in sorted(large_arrays):
        if is_norm_tensor(base_name):
            continue
        if is_protected_tensor(base_name):
            continue

        layers_dict = array_layers[base_name]
        metrics_dict, ref_peak, avg_val = get_score_source(base_name)

        if metrics_dict is None or ref_peak is None:
            cmd_parts.append(f"--tensor-type blk.*.{base_name}={DEFAULT_TARGET}")
            continue

        available_scores = [metrics_dict.get(ln, 0.0) for ln in layers_dict if ln in metrics_dict]
        if not available_scores:
            cmd_parts.append(f"--tensor-type blk.*.{base_name}={DEFAULT_TARGET}")
            continue

        total_expected_layers = max_layer + 1
        is_partial_coverage = len(layers_dict) < total_expected_layers

        for layer_num in layers_dict:
            score = metrics_dict.get(layer_num, 0.0)
            if layer_num not in all_layer_scores or score > all_layer_scores[layer_num]:
                all_layer_scores[layer_num] = score

        array_score_info[base_name] = (layers_dict, metrics_dict, is_partial_coverage)

    # --- Step 6b: Calculate tier thresholds using gap detection in log space ---
    import math
    tier_map = {}
    
    if all_layer_scores:
        sorted_scores = sorted(all_layer_scores.values())
        significant_scores = sorted([s for s in sorted_scores if s > 0.01])
        n = len(sorted_scores)
        
        tier_thresholds = []
        
        if len(significant_scores) >= 4:
            log_scores = sorted([math.log(s) for s in significant_scores])
            
            all_gaps = []
            for i in range(1, len(log_scores)):
                gap_size = log_scores[i] - log_scores[i-1]
                if i >= 2 and (len(log_scores) - i) >= 2:
                    all_gaps.append((gap_size, log_scores[i]))
            
            all_gaps_sorted = sorted(all_gaps, reverse=True)
            
            if len(all_gaps_sorted) >= 3:
                boundary_logs = sorted([g[1] for g in all_gaps_sorted[:3]])
                tier_thresholds = [math.exp(b) for b in boundary_logs]
            elif len(all_gaps_sorted) >= 2:
                boundary_logs = sorted([g[1] for g in all_gaps_sorted[:2]])
                segments = [
                    [v for v in log_scores if v < boundary_logs[0]],
                    [v for v in log_scores if boundary_logs[0] <= v < boundary_logs[1]],
                    [v for v in log_scores if v >= boundary_logs[1]]
                ]
                largest_seg_idx = max(range(3), key=lambda i: len(segments[i]))
                if len(segments[largest_seg_idx]) >= 2:
                    mid_point = segments[largest_seg_idx][len(segments[largest_seg_idx]) // 2]
                    boundary_logs.append(mid_point)
                    boundary_logs = sorted(boundary_logs)
                tier_thresholds = [math.exp(b) for b in boundary_logs]
            else:
                tier_thresholds = [
                    math.exp(log_scores[len(log_scores) // 4]),
                    math.exp(log_scores[len(log_scores) // 2]),
                    math.exp(log_scores[3 * len(log_scores) // 4])
                ]
        else:
            tier_thresholds = [
                sorted_scores[n // 4],
                sorted_scores[n // 2],
                sorted_scores[3 * n // 4]
            ]
        
        tier_thresholds = sorted(set(tier_thresholds))
        if len(tier_thresholds) < 3:
            median = sorted_scores[n // 2]
            while len(tier_thresholds) < 3:
                tier_thresholds.append(median * (1.5 ** (len(tier_thresholds) - 1)))
            tier_thresholds = sorted(tier_thresholds[:3])

        for layer_num, score in all_layer_scores.items():
            if score >= tier_thresholds[2]:
                tier_map[layer_num] = 'T1'
            elif score >= tier_thresholds[1]:
                tier_map[layer_num] = 'T2'
            elif score >= tier_thresholds[0]:
                tier_map[layer_num] = 'T3'
            else:
                tier_map[layer_num] = 'T4'

    # Tier-to-target mapping
    tier_target = {
        'T1': HIGH_TARGET,
        'T2': MID_TARGET,
        'T3': LOW_TARGET,
        'T4': DEFAULT_TARGET
    }

    # --- Step 6c: Assign targets based on tiers ---
    layer_best_target = {}
    layer_best_score = {}
    estimated_total_bits = 0

    for base_name, (layers_dict, metrics_dict, is_partial_coverage) in sorted(array_score_info.items()):
        if is_partial_coverage:
            best_target_for_array = DEFAULT_TARGET
            for layer_num in sorted(layers_dict.keys()):
                elem_count = layers_dict[layer_num]
                score = metrics_dict.get(layer_num, 0.0)

                tier = tier_map.get(layer_num, 'T4')
                assigned_target = tier_target[tier]

                if layer_num in TAPERED_TARGETS:
                    assigned_target = get_best_target(assigned_target, TAPERED_TARGETS[layer_num])
                elif layer_num == END_LAYER:
                    assigned_target = get_best_target(assigned_target, END_TARGET)

                best_target_for_array = get_best_target(best_target_for_array, assigned_target)
                estimated_total_bits += elem_count * BPW_MAP.get(assigned_target, 2.06)

                if layer_num not in layer_best_target:
                    layer_best_target[layer_num] = assigned_target
                    layer_best_score[layer_num] = score
                else:
                    layer_best_target[layer_num] = get_best_target(layer_best_target[layer_num], assigned_target)
                    layer_best_score[layer_num] = max(layer_best_score[layer_num], score)

            cmd_parts.append(f"--tensor-type blk.*.{base_name}={best_target_for_array}")
        else:
            for layer_num in sorted(layers_dict.keys()):
                elem_count = layers_dict[layer_num]
                score = metrics_dict.get(layer_num, 0.0)

                tier = tier_map.get(layer_num, 'T4')
                assigned_target = tier_target[tier]

                if layer_num in TAPERED_TARGETS:
                    assigned_target = get_best_target(assigned_target, TAPERED_TARGETS[layer_num])
                elif layer_num == END_LAYER:
                    assigned_target = get_best_target(assigned_target, END_TARGET)

                if assigned_target != DEFAULT_TARGET:
                    cmd_parts.append(f"--tensor-type blk.{layer_num}.{base_name}={assigned_target}")
                estimated_total_bits += elem_count * BPW_MAP.get(assigned_target, 2.06)

                if layer_num not in layer_best_target:
                    layer_best_target[layer_num] = assigned_target
                    layer_best_score[layer_num] = score
                else:
                    layer_best_target[layer_num] = get_best_target(layer_best_target[layer_num], assigned_target)
                    layer_best_score[layer_num] = max(layer_best_score[layer_num], score)

    # Collect ALL applicable targets per layer and pick the finest using get_best_target()
    small_tiny_ffn_layers = {}
    for base_name in sorted(small_arrays + tiny_arrays):
        if is_norm_tensor(base_name) or is_protected_tensor(base_name):
            continue
        lower = base_name.lower()
        if not lower.startswith('ffn_'):
            continue
        layers_dict = array_layers[base_name]
        target = COPY_TARGET if base_name in small_arrays else TINY_TARGET
        size_class = "SMALL" if base_name in small_arrays else "TINY"
        for layer_num in layers_dict:
            score = layer_ffn_metrics.get(layer_num, 0.0)
            if layer_num not in small_tiny_ffn_layers:
                small_tiny_ffn_layers[layer_num] = (target, score, size_class)
            else:
                existing = small_tiny_ffn_layers[layer_num]
                if score > existing[1]:
                    small_tiny_ffn_layers[layer_num] = (target, score, size_class)

    tapered_end_targets = {}
    for layer_num, target in TAPERED_TARGETS.items():
        tapered_end_targets[layer_num] = target
    tapered_end_targets[END_LAYER] = END_TARGET

    # Only fill gaps for layers NOT already assigned by scored large arrays.
    # This prevents small/tiny array targets from overwriting computed tier assignments.
    for layer_num, (target, score, size_class) in small_tiny_ffn_layers.items():
        if layer_num not in layer_best_target:
            layer_best_target[layer_num] = target
            layer_best_score[layer_num] = score

    # Tapered/end targets only fill gaps too - tiers from large arrays take priority.
    for layer_num, target in tapered_end_targets.items():
        score = layer_ffn_metrics.get(layer_num, 0.0)
        if layer_num not in layer_best_target:
            layer_best_target[layer_num] = target
            layer_best_score[layer_num] = score

    # Build layer_assignments from aggregated per-layer data (one entry per layer)
    all_displayed_layers = set(layer_best_target.keys())
    layer_assignments = []
    for layer_num in sorted(all_displayed_layers):
        layer_ffn_types = layer_types.get(f"{layer_num}_ffn", set())
        has_exps = "_exps" in layer_ffn_types
        has_dense = "dense" in layer_ffn_types

        layer_has_exps_tensors = False
        layer_has_dense_tensors = False
        for tensor_name in tensor_element_counts:
            match = re.match(r'blk\.' + str(layer_num) + r'\.(.+)', tensor_name)
            if match:
                tname = match.group(1).lower()
                if 'exps' in tname and tname.startswith('ffn_'):
                    layer_has_exps_tensors = True
                elif tname.startswith('ffn_gate.weight') or tname.startswith('ffn_down.weight') or tname.startswith('ffn_up.weight'):
                    if 'exps' not in tname:
                        layer_has_dense_tensors = True

        effective_exps = has_exps or layer_has_exps_tensors
        effective_dense = has_dense or layer_has_dense_tensors

        if effective_exps and effective_dense:
            display_type = "Hybrid"
        elif effective_exps:
            display_type = "MoE Expert"
        elif effective_dense:
            display_type = "Dense FFN"
        else:
            any_exps = any("_exps" in v for v in layer_types.values())
            display_type = "MoE Expert" if any_exps else "Dense FFN"

        assigned = layer_best_target.get(layer_num, DEFAULT_TARGET)
        val = layer_best_score.get(layer_num, 0.0)

        # Display tier logic (generic for all architectures):
        # 1. TAPERED/END override if the layer is in those special sets
        # 2. Computed tier from imatrix score gap detection (works for dense/MoE/hybrid)
        # 3. Size class fallback only when no computed tier exists
        if layer_num in TAPERED_TARGETS:
            tier = "TAPERED"
        elif layer_num == END_LAYER and layer_num in tapered_end_targets:
            tier = "END"
        else:
            t = tier_map.get(layer_num, None)
            if t is not None:
                tier = t
            elif layer_num in small_tiny_ffn_layers:
                tier = small_tiny_ffn_layers[layer_num][2]
            else:
                tier = 'T4'

        layer_assignments.append({
            'layer': layer_num,
            'display_type': display_type,
            'val': val,
            'assigned': assigned,
            'has_ssm': layer_num in layer_ssm_metrics,
            'tier': tier
        })

    peak_score = max(v['val'] for v in layer_assignments) if layer_assignments else 0.0

    return {
        'cmd_parts': cmd_parts,
        'estimated_total_bits': estimated_total_bits,
        'layer_assignments': layer_assignments,
        'peak_score': peak_score,
        'default_target': DEFAULT_TARGET
    }


# ==============================================================================
# YMQ Stage 3: Visualization & Size Estimation
# ==============================================================================
def ymq_stage3_visualize(data, target_data):
    """Render layer profiler table, array summary, per-target breakdown, and final size estimate."""
    
    tensor_element_counts = data['tensor_element_counts']
    layer_ffn_metrics = data['layer_ffn_metrics']
    layer_ssm_metrics = data['layer_ssm_metrics']
    layer_dimensions = data['layer_dimensions']
    max_layer = data['max_layer']
    imatrix_data = data['imatrix_data']
    cmd_parts = target_data['cmd_parts']
    layer_assignments = target_data['layer_assignments']
    peak_score = target_data['peak_score']
    default_target = target_data['default_target']
    
    print("=" * 75, file=sys.stderr)
    print(f" {'LAYER':^7} | {'ARCH TYPE':^12} | {'IMATRIX SCORE':^18} | {'TIER':^6} | {'ASSIGNED TARGET':^15}", file=sys.stderr)
    print("=" * 75, file=sys.stderr)
    
    for assignment in layer_assignments:
        layer = assignment['layer']
        display_type = assignment['display_type']
        val = assignment['val']
        assigned = assignment['assigned']
        tier = assignment.get('tier', 'T4')
        
        if isinstance(layer, int):
            layer_label = f"L{layer:02d}"
        else:
            layer_str = str(layer)
            if "_ssm" in layer_str:
                num_part = layer_str.replace("_ssm", "")
                layer_label = f"L{int(num_part):02d}_ssm"
            else:
                layer_label = f"L{layer_str}"
        
        print(f"  {layer_label:<8}| {display_type:^12} | {val:^18.6f} | {tier:^6} | {assigned:^15}", file=sys.stderr)
    
    print("=" * 75, file=sys.stderr)
    
    tensor_assignments = {}
    for arg in cmd_parts:
        if arg.startswith("--tensor-type "):
            tensor_type = arg.replace("--tensor-type ", "").split("=")
            if len(tensor_type) == 2:
                name, target = tensor_type[0], tensor_type[1]
                tensor_assignments[name] = target
    
    def get_base_array_name(tensor_name):
        if "blk.*." in tensor_name:
            return tensor_name.replace("blk.*.", "")
        elif tensor_name.startswith("blk."):
            return re.sub(r'^blk\.\d+\.', '', tensor_name)
        return tensor_name
    
    def get_array_weights(base_name):
        total_elements = 0
        found_layers = 0
        
        for tensor_name, elem_count in tensor_element_counts.items():
            cleaned = get_base_array_name(tensor_name)
            if cleaned == base_name:
                total_elements += elem_count
                found_layers += 1
        
        if found_layers > 0:
            return total_elements // found_layers
        return 0
    
    array_layer_targets = defaultdict(dict)
    array_base_targets = {}
    
    for tensor_name, target in tensor_assignments.items():
        base_name = get_base_array_name(tensor_name)
        
        if tensor_name.startswith("blk."):
            if "blk.*" in tensor_name:
                array_base_targets[base_name] = target
            else:
                layer_num = int(tensor_name.split("blk.")[1].split(".")[0])
                array_layer_targets[base_name][layer_num] = target
    
    total_array_bits = 0
    array_display = {}
    
    for base_name in set(list(array_base_targets.keys()) + list(array_layer_targets.keys())):
        default_target_local = array_base_targets.get(base_name, default_target)
        
        weights_per_layer = get_array_weights(base_name)
        num_layers_with_override = len(array_layer_targets.get(base_name, {}))
        
        actual_layer_count = 0
        for tensor_name in tensor_element_counts:
            cleaned = get_base_array_name(tensor_name)
            if cleaned == base_name:
                actual_layer_count += 1
        
        num_layers_default = actual_layer_count - num_layers_with_override
        
        total_bits = 0
        
        default_bpw = BPW_MAP.get(default_target_local, 4.0)
        if weights_per_layer > 0 and num_layers_default > 0:
            total_bits += num_layers_default * weights_per_layer * default_bpw
        
        for layer_num, override_target in array_layer_targets.get(base_name, {}).items():
            override_bpw = BPW_MAP.get(override_target, 4.0)
            layer_key = f"{layer_num}_{base_name}"
            actual_size = layer_dimensions.get(layer_key, weights_per_layer)
            total_bits += actual_size * override_bpw
        
        if num_layers_with_override == 0:
            display_target = default_target_local
        elif num_layers_default == 0:
            display_target = next(iter(array_layer_targets[base_name].values()))
        else:
            display_target = default_target_local + " (partial)"
        
        total_elements_for_q8 = weights_per_layer * num_layers_default if weights_per_layer > 0 and num_layers_default > 0 else 0
        for layer_num, override_target in array_layer_targets.get(base_name, {}).items():
            layer_key = f"{layer_num}_{base_name}"
            actual_size = layer_dimensions.get(layer_key, weights_per_layer)
            total_elements_for_q8 += actual_size
        
        q8_bits = total_elements_for_q8 * BPW_MAP["Q8_0"] if total_elements_for_q8 > 0 else 0
        
        avg_score = 0.0
        if "ffn_down" in base_name or "ffn_gate" in base_name or "ffn_up" in base_name:
            ffn_scores = [layer_ffn_metrics.get(l, 0.0) for l in range(max_layer + 1)]
            avg_score = sum(ffn_scores) / len(ffn_scores) if ffn_scores else 0.0
        elif "ssm_out" in base_name or "ssm_in" in base_name:
            ssm_scores = [layer_ssm_metrics.get(l, 0.0) for l in range(max_layer + 1)]
            avg_score = sum(ssm_scores) / len(ssm_scores) if ssm_scores else 0.0
        
        actual_layer_count = num_layers_with_override + num_layers_default
        array_display[base_name] = (actual_layer_count, total_bits, display_target, q8_bits, avg_score)
        total_array_bits += total_bits
    
    print("", file=sys.stderr)
    print("=" * 115, file=sys.stderr)
    print(f" {'ARRAY NAME':^30} | {'LAYERS':^7} | {'Q8 SIZE (GB)':^15} | {'ESTIMATED (GB)':^15} | {'TARGET':^15} | {'IMATRIX AVG':^15}", file=sys.stderr)
    print("=" * 115, file=sys.stderr)
    
    for array_name, (num_layers, total_bits, display_target, q8_bits, avg_score) in sorted(array_display.items(), key=lambda x: x[1][3], reverse=True):
        q8_gb = q8_bits / (8.0 * 1024.0 * 1024.0 * 1024.0)
        est_gb = total_bits / (8.0 * 1024.0 * 1024.0 * 1024.0)
        print(f" {array_name:^30} | {num_layers:^7} | {q8_gb:^15.4f} | {est_gb:^15.4f} | {display_target:^15} | {avg_score:^15.6f}", file=sys.stderr)
    
    print("=" * 115, file=sys.stderr)
    
    wildcard_targets = {}
    specific_targets = {}
    
    for arg in cmd_parts:
        if arg.startswith("--tensor-type "):
            rest = arg.replace("--tensor-type ", "")
            eq_idx = rest.index("=")
            pattern = rest[:eq_idx]
            target = rest[eq_idx+1:]
            
            if pattern.startswith("blk.*."):
                base_name = pattern.replace("blk.*.", "")
                wildcard_targets[base_name] = target
            elif pattern.startswith("blk."):
                parts = pattern.split(".")
                layer_num = int(parts[1])
                base_name = ".".join(parts[2:])
                specific_targets[(layer_num, base_name)] = target
    
    def resolve_tensor_target(tensor_name):
        """Resolve the quantization target for a given tensor name."""
        if not tensor_name.startswith("blk."):
            return tensor_assignments.get(tensor_name, default_target)
        
        match = re.match(r'blk\.(\d+)\.(.+)', tensor_name)
        if not match:
            return default_target
        
        layer_num = int(match.group(1))
        base_name = match.group(2)
        
        # Check specific (per-layer) targets first, then wildcard targets
        if (layer_num, base_name) in specific_targets:
            return specific_targets[(layer_num, base_name)]
        elif base_name in wildcard_targets:
            return wildcard_targets[base_name]
        return default_target

    # Calculate total bits and per-target breakdown in a single pass
    target_bits = defaultdict(float)
    total_bits_from_tensors = 0
    
    for tensor_name, n_elements in tensor_element_counts.items():
        assigned_target = resolve_tensor_target(tensor_name)
        bpw = BPW_MAP.get(assigned_target, 4.0)
        bits = n_elements * bpw
        total_bits_from_tensors += bits
        target_bits[assigned_target] += bits
    
    overhead_factor = 1.06 if b"exps" in imatrix_data else 1.05
    estimated_gb = (total_bits_from_tensors * overhead_factor) / (8.0 * 1024.0 * 1024.0 * 1024.0)
    
    print("", file=sys.stderr)
    print("=" * 55, file=sys.stderr)
    print(f" {'TARGET':^15} | {'ESTIMATED (GB)':^15}", file=sys.stderr)
    print("=" * 55, file=sys.stderr)
    
    for target in sorted(target_bits.keys(), key=lambda t: target_bits[t], reverse=True):
        gb = target_bits[target] / (8.0 * 1024.0 * 1024.0 * 1024.0)
        print(f" {target:^15} | {gb:^15.4f}", file=sys.stderr)
    
    print("=" * 55, file=sys.stderr)
    
    gguf_path = data.get('gguf_path', '')
    
    print(f"--------------------------------------------------------", file=sys.stderr)
    if gguf_path and os.path.exists(gguf_path):
        source_size = os.path.getsize(gguf_path) / (1024.0 * 1024.0 * 1024.0)
        print(f" SOURCE Q8_0 FILE SIZE: ~{source_size:.2f}GB", file=sys.stderr)
        if estimated_gb < source_size:
            savings = ((source_size - estimated_gb) / source_size) * 100
            print(f" ESTIMATED SIZE SAVINGS: ~{savings:.1f}%", file=sys.stderr)
    print(f" ESTIMATED FINAL QUANTIZED FILE SIZE: ~{estimated_gb:.2f}GB", file=sys.stderr)
    print(f"--------------------------------------------------------", file=sys.stderr)


# ==============================================================================
# Main Execution
# ==============================================================================

imatrix_path = "${IMATRIX_PATH}"
gguf_path = "${SOURCE_GGUF}"

input_target = sys.argv[1] if len(sys.argv) > 1 else "Q4_K_M"
high_target = sys.argv[2] if len(sys.argv) > 2 else "Q5_K"
mid_target = sys.argv[3] if len(sys.argv) > 3 else "IQ4_XS"
low_target = sys.argv[4] if len(sys.argv) > 4 else "IQ3_S"
copy_target = sys.argv[5] if len(sys.argv) > 5 else "COPY"
tiny_target = sys.argv[6] if len(sys.argv) > 6 else "COPY"
default_target = sys.argv[7] if len(sys.argv) > 7 else "IQ2_XXS"
small_threshold_gb = sys.argv[8] if len(sys.argv) > 8 else "1.0"
tiny_threshold_gb = sys.argv[9] if len(sys.argv) > 9 else "0.1"

data = ymq_stage1_analysis(imatrix_path, gguf_path)
target_data = ymq_stage2_assign_targets(data, input_target, high_target, mid_target, low_target, copy_target, tiny_target, default_target, small_threshold_gb, tiny_threshold_gb)
ymq_stage3_visualize(data, target_data)

for arg in target_data['cmd_parts']:
    print("    " + arg + " \\\\")
EOF
)

# Build the executable file safely using the pristine stdout capture
mkdir -p "$OUTPUT_DIR"
cat << EOF > "${OUTPUT_DIR}/run_quant.sh"
#!/usr/bin/env bash
./llama-quantize \\
    --imatrix "${IMATRIX_PATH}" \\
${QUANT_ARGS}
    "${SOURCE_GGUF}" \\
    "${OUTPUT_GGUF}" \\
    ${DEFAULT_TARGET}
EOF

chmod +x "${OUTPUT_DIR}/run_quant.sh"
echo "YMQ-Compiler: Quantization script compiled to ${OUTPUT_DIR}/run_quant.sh"
