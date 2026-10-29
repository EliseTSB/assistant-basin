# bench.sh -- time the same two A770 jobs on a rented RTX 3090 (Claude, 2026-10-29; Elise asked how much a 3090 would help)
cd /root/bench
date +%s > run.start
python -m pip install -q transformers==5.13.0 accelerate huggingface_hub pandas pyarrow > pip.log 2>&1; echo pip=$? >> run.log
python dl.py allenai/OLMo-2-0425-1B-Instruct 48d788eca847d4d7548f375ad03d3c9312f6139e > olmo.path 2>> dl.log; echo dl=$? >> run.log
python -c "from huggingface_hub import hf_hub_download as h; print(h('Salesforce/wikitext','wikitext-103-raw-v1/train-00000-of-00002.parquet',repo_type='dataset',local_dir='/root/wt'))" > wt.path 2>> dl.log; echo dl_wt=$? >> run.log
sha256sum "$(tail -1 wt.path)" >> run.log
M="$(tail -1 olmo.path)"
python -c "import torch,transformers;print(torch.__version__,transformers.__version__,torch.cuda.get_device_name(0))" > versions.txt 2>&1
# 1. screen, exactly the A770 command for layer 3 (8,448 rows in 433.6 s there, batch 48, 40 new tokens)
echo "screen_b48 start $(date +%s.%N)" >> run.log; python -u part92/screen92.py --model "$M" --layers 3 --alpha 0.8 --random 256 --max-new 40 --batch 48 --out res/screen-b48 --device cuda > screen-b48.log 2>&1; echo "screen_b48 rc=$? end $(date +%s.%N)" >> run.log >> run.log
# 2. the same at batch 192, what 24 GB allows
echo "screen_b192 start $(date +%s.%N)" >> run.log; python -u part92/screen92.py --model "$M" --layers 3 --alpha 0.8 --random 256 --max-new 40 --batch 192 --out res/screen-b192 --device cuda > screen-b192.log 2>&1; echo "screen_b192 rc=$? end $(date +%s.%N)" >> run.log >> run.log
# 3. lens fit, 10 prompts, seed 9801, dim_batch 8 (A770: about 38 s per prompt)
echo "fit start $(date +%s.%N)" >> run.log; python -u fit_jlens_olmo.py --device cuda --seed 9801 --n-prompts 10 --data "$(tail -1 wt.path)" --cache-dir /root/hf/hub --out res/lens10.pt > fit.log 2>&1; echo "fit rc=$? end $(date +%s.%N)" >> run.log
# 4. the same at dim_batch 32
python -u fit_jlens_olmo.py --device cuda --seed 9801 --n-prompts 10 --dim-batch 32 --data "$(tail -1 wt.path)" --cache-dir /root/hf/hub --out res/lens10-db32.pt > fit-db32.log 2>&1; echo fit_db32=$? >> run.log
date +%s > run.end
echo ALLDONE >> run.log
