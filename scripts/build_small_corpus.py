from datasets import load_dataset
import os

out = "/home/iscb/wolfson/annab4/AutoSerum/data/slices.txt"
os.makedirs(os.path.dirname(out), exist_ok=True)

ds = load_dataset("cerebras/SlimPajama-627B", split="train", streaming=True)

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
