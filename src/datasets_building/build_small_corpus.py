from datasets import load_dataset
import os

from config import PATHS, BUILD_AUX

out = PATHS["corpus"]
os.makedirs(os.path.dirname(out), exist_ok=True)

dataset = BUILD_AUX["dataset"]
split = BUILD_AUX["split"]
ds = load_dataset(dataset, split=split, streaming=True)

n = 1000
with open(out, "w", encoding="utf-8") as f:
    i = 0
    for ex in ds:
        txt = ex.get("text")
        if not txt:
            continue
        # one line per example; no CRs
        f.write(txt.replace("\r", " ").strip() + "\n")
        i += 1
        if i >= n:
            break

print("OK ->", out)
