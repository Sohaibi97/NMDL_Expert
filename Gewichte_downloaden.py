import json
from huggingface_hub import hf_hub_download

repo = "mistralai/Ministral-3-3B-Base-2512"
index_path = hf_hub_download(repo, "model.safetensors.index.json")
with open(index_path, "r", encoding="utf-8") as f:
    index = json.load(f)

shards = sorted(set(index["weight_map"].values()))
print("num_shards:", len(shards))
for s in shards:
    print("downloading:", s)
    hf_hub_download(repo, s)
print("done")