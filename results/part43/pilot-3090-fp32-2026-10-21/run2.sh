cd /root/work
echo "$(date -u +%FT%TZ) start diag"; python -u /root/diag43.py; echo "diag rc=$?"
cd /root/infer
echo "$(date -u +%FT%TZ) start fastgen fp32"; HF_HUB_CACHE=/root/hf-cache python -u fastgen.py --model Qwen/Qwen3-1.7B --device cuda --cache-dir /root/hf-cache --dtype float32 --check-equal 12 > check-equal-qwen3-1.7b-fp32-3090.json.log 2>&1; echo "fastgen fp32 rc=$?"
echo "$(date -u +%FT%TZ) start fastgen bf16"; python -u fastgen.py --model Qwen/Qwen3-1.7B --device cuda --cache-dir /root/hf-cache --dtype bfloat16 --check-equal 12 > check-equal-qwen3-1.7b-bf16-3090.json.log 2>&1; echo "fastgen bf16 rc=$?"
echo "$(date -u +%FT%TZ) start bench"; python -u bench_infer.py --model Qwen/Qwen3-1.7B --device cuda --cache-dir /root/hf-cache --dtype bfloat16 --batch-sizes 1,4,8,16,32 --attn-implementations sdpa,eager --decode-tokens 32 --iterations 2 --skip-compile-smoke --out bench-qwen3-1.7b-3090.json > bench-qwen3-1.7b-3090.table.txt 2>&1; echo "bench rc=$?"
echo "$(date -u +%FT%TZ) ALL DONE"
