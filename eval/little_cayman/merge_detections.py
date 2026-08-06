"""Merge per-GPU detection shards into detections.json."""
import json, os, glob, collections
SP = os.path.dirname(__file__)
recs = []
for f in sorted(glob.glob(os.path.join(SP, "detections_shard*.json"))):
    recs.extend(json.load(open(f)))
json.dump(recs, open(os.path.join(SP, "detections.json"), "w"), indent=2)
n = len(recs); m = sum(r["matched"] for r in recs)
print(f"merged {n} recs from shards, recall {m}/{n} = {m/n:.3f}")
per = collections.defaultdict(lambda: [0, 0])
for r in recs:
    per[r["gt"]][1] += 1
    if r["matched"]: per[r["gt"]][0] += 1
for k in sorted(per, key=lambda k: -per[k][1]):
    c, t = per[k]; print(f"  {c:>3}/{t:<3} {c/t:.2f}  {k}")
