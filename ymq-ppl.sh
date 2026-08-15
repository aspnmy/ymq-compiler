#!/bin/bash
# =================================================================================
#               YMQ-COMPILER AUTOMATED PERPLEXITY EVALUATION SUITE
# =================================================================================

# 1. Configuration Settings
LLAMA_PPL_PATH="./llama-perplexity" # Adjust if your binary sits inside build/bin/
DATASET="/mnt6/models/wiki.test.raw"
MODEL_DIR="/"
OUTPUT_FILE="./ymq_perplexity_results.md"

models=(
    "mnt6/models/Qwen3.8-27B-Uncensored-YMQ-XXS.gguf"
    "mnt4/models/Qwen3.8-27B-Uncensored-YMQ-XS.gguf"
    "mnt4/models/Qwen3.8-27B-Uncensored-YMQ-M.gguf"
    "mnt4/models/Qwen3.8-27B-Uncensored-YMQ-L.gguf"
    "mnt6/models/Qwen3.8-27B-Uncensored-YMQ-XL.gguf"
)

# 2. Initialize Report File
echo "## 📉 YMQ-Compiler Perplexity (PPL) Quantization Analysis" > "$OUTPUT_FILE"
echo -e "| Model Preset Variant | File Size | WikiText-2 Perplexity | Quality Retained vs BF16 |" >> "$OUTPUT_FILE"
echo -e "| :--- | :--- | :--- | :--- |" >> "$OUTPUT_FILE"

# 3. Execution Processing Loop
for model in "${models[@]}"; do
    FULL_PATH="$MODEL_DIR/$model"
    
    if [ ! -f "$FULL_PATH" ]; then
        echo "⚠️ Skipping $model - File not found"
        continue
    fi

    echo "------------------------------------------------------------------------"
    echo "📊 Evaluating Perplexity for: $model"
    echo "------------------------------------------------------------------------"
    
    # Run the native perplexity engine pass
    # -c 4096 tests the base processing capability
    # --chunks 100 processes a clean statistical slice quickly
    PPL_OUTPUT=$($LLAMA_PPL_PATH \
        -m "$FULL_PATH" \
        -f "$DATASET" \
        -c 4096 \
        --chunks 100 \
        -fa 1 \
        -ngl 99 \
        -t 8 \
        2>&1)

    # Parse out the final numerical perplexity score from the log trace lines
    # llama-perplexity outputs a final line containing text: "Final perplexity: X.XXXX"
    FINAL_PPL=$(echo "$PPL_OUTPUT" | grep -o 'Final estimate: PPL = [0-9.]*' | sed 's/Final estimate: PPL = //')
    
    # Pull the real physical file size in gigabytes
    FILE_SIZE=$(du -sh "$FULL_PATH" | awk '{print $1}')

    # Output line directly to your report file template
    echo "| ${model##*-YMQ-} | $FILE_SIZE | $FINAL_PPL | TBD |" >> "$OUTPUT_FILE"
    echo "✅ Completed! Score: $FINAL_PPL"
done

echo "🎉 Perplexity sweep finished! Results exported to: $OUTPUT_FILE"
