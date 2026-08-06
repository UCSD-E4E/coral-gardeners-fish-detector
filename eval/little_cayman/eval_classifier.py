"""Classifier-only eval: crop labeled fish from cached rectified frames,
run BioCLIP under two label-spaces (13-species prior vs 69-species open-set),
score top-1/top-5 vs ground-truth species.
"""
import json, os, sys, math, collections
import cv2
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
SP = os.path.dirname(__file__)
FRAMES = os.path.join(SP, "cache", "frames")
CROPS = os.path.join(SP, "cache", "crops")
os.makedirs(CROPS, exist_ok=True)

manifest = json.load(open(os.path.join(SP, "manifest.json")))
sets = json.load(open(os.path.join(SP, "species_sets.json")))
THIRTEEN = sets["little_cayman"]
OPEN = sets["open_union"]


def crop_box(ht, W, H, pad=0.4):
    xs = [ht["head_x"], ht["tail_x"]]; ys = [ht["head_y"], ht["tail_y"]]
    length = math.hypot(ht["head_x"] - ht["tail_x"], ht["head_y"] - ht["tail_y"])
    m = pad * length
    return (max(0, int(min(xs) - m)), max(0, int(min(ys) - m)),
            min(W, int(max(xs) + m)), min(H, int(max(ys) + m)))


def seg_point_dist(ht, px, py):
    ax, ay, bx, by = ht["head_x"], ht["head_y"], ht["tail_x"], ht["tail_y"]
    vx, vy = bx - ax, by - ay
    L2 = vx*vx + vy*vy
    if L2 == 0:
        return math.hypot(px-ax, py-ay)
    t = max(0, min(1, ((px-ax)*vx + (py-ay)*vy)/L2))
    return math.hypot(px-(ax+t*vx), py-(ay+t*vy))


def choose_fish(m):
    """Return the head/tail box for the labeled (laser-on) fish."""
    hts = m["headtail"]
    if not hts:
        return None
    if len(hts) == 1:
        return hts[0]
    if m["laser"]:
        lx, ly = m["laser"][0]["x"], m["laser"][0]["y"]
        return min(hts, key=lambda ht: seg_point_dist(ht, lx, ly))
    return None  # ambiguous multi-fish, no laser -> skip


# ---- build crops ----
cases = []  # (image_id, gt_sci, crop_path)
skipped = collections.Counter()
for m in manifest:
    gt = m["species"]["scientific"]
    frame = os.path.join(FRAMES, f'{m["image_id"]}.jpg')
    if not os.path.isfile(frame):
        skipped["no_frame"] += 1; continue
    ht = choose_fish(m)
    if ht is None:
        skipped["no_or_ambiguous_headtail"] += 1; continue
    crop_path = os.path.join(CROPS, f'{m["image_id"]}.jpg')
    if not os.path.isfile(crop_path):
        img = cv2.imread(frame)
        H, W = img.shape[:2]
        x0, y0, x1, y1 = crop_box(ht, W, H)
        if x1 - x0 < 8 or y1 - y0 < 8:
            skipped["tiny_crop"] += 1; continue
        cv2.imwrite(crop_path, img[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 95])
    cases.append((m["image_id"], gt, crop_path))

print(f"crops ready: {len(cases)}  skipped: {dict(skipped)}", flush=True)

# ---- classify under each label space ----
from coral_fish_pipeline.classification.bioclip_classifier import BioCLIPClassifier

HF_CACHE = os.path.join(REPO, "models", "hf_cache")
os.environ.setdefault("HF_HOME", HF_CACHE)


def run_condition(name, species_list):
    print(f"\n=== condition: {name}  ({len(species_list)} classes) ===", flush=True)
    clf = BioCLIPClassifier(species=species_list, region="little_cayman",
                            device="auto", precision="fp16",
                            unknown_threshold=0.0, uncertain_margin=0.0)
    clf.load()
    preds = []
    for i, (img_id, gt, crop_path) in enumerate(cases):
        r = clf._predict_image(crop_path)
        top5 = [s for s, _ in r["top5"]]
        preds.append(dict(image_id=img_id, gt=gt, pred=r["species"],
                          conf=r["confidence"], top5=top5,
                          top5_probs=[float(p) for _, p in r["top5"]]))
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{len(cases)}", flush=True)
    # metrics
    n = len(preds)
    top1 = sum(p["gt"] == p["pred"] for p in preds)
    top5 = sum(p["gt"] in p["top5"] for p in preds)
    per = collections.defaultdict(lambda: [0, 0])
    for p in preds:
        per[p["gt"]][1] += 1
        if p["gt"] == p["pred"]:
            per[p["gt"]][0] += 1
    out = dict(condition=name, n=n, top1=top1, top5=top5,
               top1_acc=top1/n, top5_acc=top5/n,
               per_species={k: {"correct": v[0], "n": v[1], "acc": v[0]/v[1]} for k, v in per.items()},
               preds=preds)
    json.dump(out, open(os.path.join(SP, f"results_classifier_{name}.json"), "w"), indent=2)
    print(f"  top1 {top1}/{n} = {top1/n:.3f}   top5 {top5}/{n} = {top5/n:.3f}", flush=True)
    for k in sorted(per, key=lambda k: -per[k][1]):
        c, t = per[k]
        print(f"    {c:>3}/{t:<3} {c/t:5.2f}  {k}", flush=True)
    return out


run_condition("prior13", THIRTEEN)
run_condition("open69", OPEN)
print("\nALL CONDITIONS DONE", flush=True)
