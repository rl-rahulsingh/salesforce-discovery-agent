"""
inspect_index.py — read index.pkl and report its exact structure.

Run once before writing the migration, so the migration is written
against reality rather than assumption.
"""

import pickle
import os

INDEX_PATH = os.path.join("docs", "index.pkl")

with open(INDEX_PATH, "rb") as f:
    index = pickle.load(f)

print("=" * 60)
print("TOP-LEVEL KEYS")
print("=" * 60)
for key in index:
    value = index[key]
    kind = type(value).__name__
    size = len(value) if hasattr(value, "__len__") else "n/a"
    print(f"  {key!r:20} type={kind:12} len={size}")

print()
print("=" * 60)
print("FIRST CHUNK — every field and its type")
print("=" * 60)
first = index["chunks"][0]
print(f"  chunk object type: {type(first).__name__}")
if isinstance(first, dict):
    for k, v in first.items():
        preview = str(v)[:70].replace("\n", " ")
        print(f"  {k!r:20} = {preview!r}")
else:
    print(f"  {first}")

print()
print("=" * 60)
print("VECTORS")
print("=" * 60)
vecs = index["vectors"]
print(f"  container type : {type(vecs).__name__}")
print(f"  count          : {len(vecs)}")
print(f"  one vector type: {type(vecs[0]).__name__}")
print(f"  dimensions     : {len(vecs[0])}")
print(f"  first 5 values : {list(vecs[0][:5])}")

print()
print("=" * 60)
print("DISTINCT FILENAMES")
print("=" * 60)
names = sorted({c["filename"] for c in index["chunks"]})
for n in names:
    count = sum(1 for c in index["chunks"] if c["filename"] == n)
    print(f"  {n:25} {count} chunks")