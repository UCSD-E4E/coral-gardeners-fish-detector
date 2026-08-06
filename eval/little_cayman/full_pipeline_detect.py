"""Full-pipeline stage A: SAM3 detect on rectified frames, match to the labeled
fish (head/tail midpoint containment), save matched detection crops, record recall.
"""
import json, os, sys, math, collections
import cv2
import numpy as np
from PIL import Image

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
SP = os.path.dirname(__file__)
FRAMES = os.path.join(SP, "cache", "frames")
DET_CROPS = os.path.join(SP, "cache", "det_crops")
MASKDIR = os.path.join(SP, "cache", "sam3_masks")
os.makedirs(DET_CROPS, exist_ok=True); os.makedirs(MASKDIR, exist_ok=True)
os.environ.setdefault("HF_HOME", os.path.join(REPO, "models", "hf_cache"))

manifest = json.load(open(os.path.join(SP, "manifest.json")))


def seg_point_dist(ht, px, py):
    ax, ay, bx, by = ht["head_x"], ht["head_y"], ht["tail_x"], ht["tail_y"]
    vx, vy = bx-ax, by-ay; L2 = vx*vx+vy*vy
    if L2 == 0: return math.hypot(px-ax, py-ay)
    t = max(0, min(1, ((px-ax)*vx+(py-ay)*vy)/L2))
    return math.hypot(px-(ax+t*vx), py-(ay+t*vy))


def choose_fish(m):
    hts = m["headtail"]
    if not hts: return None
    if len(hts) == 1: return hts[0]
    if m["laser"]:
        lx, ly = m["laser"][0]["x"], m["laser"][0]["y"]
        return min(hts, key=lambda ht: seg_point_dist(ht, lx, ly))
    return None


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1-ix0), max(0, iy1-iy0)
    inter = iw*ih
    if inter == 0: return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0


# build test cases (same 351 as classifier-only)
cases = []
for m in manifest:
    frame = os.path.join(FRAMES, f'{m["image_id"]}.jpg')
    if not os.path.isfile(frame): continue
    ht = choose_fish(m)
    if ht is None: continue
    cases.append((m, ht))

# optional GPU sharding: SHARD/NSHARDS select a slice; pin GPU via CUDA_VISIBLE_DEVICES externally
SHARD = int(os.environ.get("SHARD", "0"))
NSHARDS = int(os.environ.get("NSHARDS", "1"))
suffix = "" if NSHARDS == 1 else f"_shard{SHARD}"
if NSHARDS > 1:
    cases = cases[SHARD::NSHARDS]
print(f"test cases: {len(cases)} (shard {SHARD}/{NSHARDS})", flush=True)

from coral_fish_pipeline.segmentation.sam3_runner import SAM3Runner
runner = SAM3Runner(min_confidence=0.25, max_detections_per_image=100,
                    nms_iou_threshold=0.40, use_tiling=False)

PAD = 0.20
recs = []
n_match = 0
for i, (m, ht) in enumerate(cases):
    frame = os.path.join(FRAMES, f'{m["image_id"]}.jpg')
    pil = Image.open(frame).convert("RGB")
    W, H = pil.size
    dets = runner.predict(pil, image_id=str(m["image_id"]), output_mask_dir=MASKDIR,
                          prompts=["fish", "small fish"])
    cx = (ht["head_x"]+ht["tail_x"])/2; cy = (ht["head_y"]+ht["tail_y"])/2
    # gt rough box from head/tail extent
    gtb = [min(ht["head_x"], ht["tail_x"]), min(ht["head_y"], ht["tail_y"]),
           max(ht["head_x"], ht["tail_x"]), max(ht["head_y"], ht["tail_y"])]
    # detections containing the fish center, not spanning whole frame
    frame_area = W*H
    containing = []
    for d in dets:
        b = d.bbox_xyxy
        if b[0] <= cx <= b[2] and b[1] <= cy <= b[3] and (b[2]-b[0])*(b[3]-b[1]) < 0.9*frame_area:
            containing.append(d)
    matched = max(containing, key=lambda d: d.score) if containing else None
    rec = dict(image_id=m["image_id"], gt=m["species"]["scientific"],
               n_dets=len(dets), matched=matched is not None,
               match_score=(float(matched.score) if matched else None),
               match_iou=(iou(matched.bbox_xyxy, gtb) if matched else 0.0),
               det_crop=None)
    if matched is not None:
        n_match += 1
        b = matched.bbox_xyxy
        bw, bh = b[2]-b[0], b[3]-b[1]
        x0 = max(0, int(b[0]-PAD*bw)); y0 = max(0, int(b[1]-PAD*bh))
        x1 = min(W, int(b[2]+PAD*bw)); y1 = min(H, int(b[3]+PAD*bh))
        img = cv2.imread(frame)
        cp = os.path.join(DET_CROPS, f'{m["image_id"]}.jpg')
        cv2.imwrite(cp, img[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 95])
        rec["det_crop"] = cp
    recs.append(rec)
    if (i+1) % 25 == 0:
        print(f"  [{i+1}/{len(cases)}] matched {n_match} recall {n_match/(i+1):.3f}", flush=True)

json.dump(recs, open(os.path.join(SP, f"detections{suffix}.json"), "w"), indent=2)
recall = n_match/len(cases) if cases else 0.0
print(f"\nDETECTION DONE: recall {n_match}/{len(cases)} = {recall:.3f}", flush=True)
# recall per species
per = collections.defaultdict(lambda: [0, 0])
for r in recs:
    per[r["gt"]][1] += 1
    if r["matched"]: per[r["gt"]][0] += 1
print("recall per species:")
for k in sorted(per, key=lambda k: -per[k][1]):
    c, t = per[k]; print(f"  {c:>3}/{t:<3} {c/t:5.2f}  {k}", flush=True)
