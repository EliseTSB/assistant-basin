cd /root/p43
python -u part43.py pilot --device cuda --cache-dir /root/hf/hub --out results/part43-pilot-3090 > pilot.log 2>&1
echo rc=$? >> pilot.log
date +%s > run.end
