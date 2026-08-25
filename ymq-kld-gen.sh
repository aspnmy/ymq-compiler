./llama-perplexity \
  -m /work/models/Qwen3.8-27B-Uncensored-f16.gguf \
  -f /mnt6/models/wiki.test.raw \
  -c 2048 \
  -b 64 \
  -ngl 99 \
  --chunks 50 \
  --save-all-logits /work/models/base_probs.kld
