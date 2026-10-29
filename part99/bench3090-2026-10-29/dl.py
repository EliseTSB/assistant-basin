import sys
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], cache_dir='/root/hf/hub',
                        allow_patterns=['*.json', '*.safetensors'], ignore_patterns=['original/*']))
