#!/bin/bash
# =================================================================================
#               YMQ-COMPILER AUTOMATED KL-DIVERGENCE EVALUATION SUITE
# =================================================================================

# 1. Configuration Settings
LLAMA_PPL_PATH="./llama-perplexity" # Adjust if your binary sits inside build/bin/
DATASET="/mnt6/models/wiki.test.raw"
MODEL_DIR="/"
OUTPUT_FILE="./ymq_kld_results.md"

models=(
    "work/models/Qwen3.8-27B-Uncensored-YMQ-S-Pro.gguf"
    "mnt4/models/Qwen3.8-27B-Unleashed-UD-IQ3_XXS.gguf"
    "mnt/models/Qwen3.8-27B-Unleashed-UD-Q3_K_XL.gguf"
)

# 2. Initialize Report File
echo "## 📉 YMQ-Compiler KL-Divergence (KLD) Quantization Analysis" > "$OUTPUT_FILE"
echo -e "| Model Preset Variant | File Size | WikiText-2 KL-Divergence | Quality Retained vs BF16 |" >> "$OUTPUT_FILE"
echo -e "| :--- | :--- | :--- | :--- |" >> "$OUTPUT_FILE"

# 3. Execution Processing Loop
for model in "${models[@]}"; do
    FULL_PATH="$MODEL_DIR/$model"
    
    if [ ! -f "$FULL_PATH" ]; then
        echo "⚠️ Skipping $model - File not found"
        continue
    fi

    echo "------------------------------------------------------------------------"
    echo "📊 Evaluating KL-Divergence for: $model"
    echo "------------------------------------------------------------------------"
    
    # Run the native KL-divergence engine pass
    # -c 4096 tests the base processing capability
    # -b 512 sets batch size for efficient computation
    $LLAMA_PPL_PATH \
        -m "$FULL_PATH" \
        -f "$DATASET" \
        -c 2048 \
        -b 64 \
        -ngl 9999 \
        --chunks 50 \
        --kl-divergence-base "/work/models/base_probs.kld" \
        --kl-divergence

    # Parse out the final numerical KL-divergence score from the log trace lines
    # llama-perplexity outputs a final line containing text: "Final estimate: KLD = X.XXXX"
    FINAL_KLD=$(echo "$KLD_OUTPUT" | grep -o 'Final estimate: KLD = [0-9.]*' | sed 's/Final estimate: KLD = //')
    
    # Pull the real physical file size in gigabytes
    FILE_SIZE=$(du -sh "$FULL_PATH" | awk '{print $1}')

    # Output line directly to your report file template
    echo "| ${model##*-YMQ-} | $FILE_SIZE | $FINAL_KLD | TBD |" >> "$OUTPUT_FILE"
    echo "✅ Completed! KLD Score: $FINAL_KLD"
done

echo "🎉 KL-Divergence sweep finished! Results exported to: $OUTPUT_FILE"