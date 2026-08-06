"""Parallel: rectify all test ORFs once, cache full rectified frames as JPEG,
emit QA overlay thumbnails for a subset first, write a cache index.

Resumable: skips images whose rectified frame is already cached.
"""
import json, os, sys, math, time
import multiprocessing as mp
import cv2
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from rectify_lib import load_raw_bgr, rectify, load_intrinsics

SP = os.path.dirname(__file__)
CACHE = os.path.join(SP, "cache")
FRAMES = os.path.join(CACHE, "frames")   # full rectified frames
QA = os.path.join(CACHE, "qa")           # annotated overlay thumbnails
os.makedirs(FRAMES, exist_ok=True)
os.makedirs(QA, exist_ok=True)

manifest = json.load(open(os.path.join(SP, "manifest.json")))
intr = load_intrinsics(os.path.join(SP, "gt", "cam_intrinsics.tsv"))
NWORKERS = int(os.environ.get("NWORKERS", "32"))

# choose QA subset: up to 2 single-fish per species -> render overlays for these first
by_sp = {}
for m in manifest:
    if m["n_fish"] == 1:
        by_sp.setdefault(m["species"]["scientific"], []).append(m)
qa_ids = set()
for sp, items in by_sp.items():
    items.sort(key=lambda m: (0 if m["laser"] else 1))
    for m in items[:2]:
        qa_ids.add(m["image_id"])


def crop_box(ht, W, H, pad=0.4):
    xs = [ht["head_x"], ht["tail_x"]]; ys = [ht["head_y"], ht["tail_y"]]
    length = math.hypot(ht["head_x"] - ht["tail_x"], ht["head_y"] - ht["tail_y"])
    m = pad * length
    return (max(0, int(min(xs) - m)), max(0, int(min(ys) - m)),
            min(W, int(max(xs) + m)), min(H, int(max(ys) + m)))


def process(m):
    t0 = time.time()
    frame_path = os.path.join(FRAMES, f'{m["image_id"]}.jpg')
    if os.path.isfile(frame_path):
        return (m["image_id"], "cached", 0.0)
    try:
        K, D = intr[m["camera_id"]]
        bgr = load_raw_bgr(m["orf_path"])
        rect = rectify(bgr, K, D)
        H, W = rect.shape[:2]
        cv2.imwrite(frame_path, rect, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if m["image_id"] in qa_ids:
            vis = rect.copy()
            for ht in m["headtail"]:
                hp = (int(ht["head_x"]), int(ht["head_y"])); tp = (int(ht["tail_x"]), int(ht["tail_y"]))
                cv2.line(vis, hp, tp, (0, 255, 255), 6)
                cv2.circle(vis, hp, 22, (0, 255, 0), -1)
                cv2.circle(vis, tp, 22, (0, 0, 255), -1)
                x0, y0, x1, y1 = crop_box(ht, W, H)
                cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 0, 255), 4)
            for L in m["laser"]:
                cv2.circle(vis, (int(L["x"]), int(L["y"])), 18, (0, 165, 255), 3)
            lab = f'{m["species"]["common"]} img{m["image_id"]} cam{m["camera_id"]} {W}x{H}'
            cv2.putText(vis, lab, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255,255,255), 6)
            cv2.putText(vis, lab, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,0,0), 2)
            sc = 720.0 / W
            cv2.imwrite(os.path.join(QA, f'thumb_{m["species"]["scientific"].replace(" ","_")}_{m["image_id"]}.jpg'),
                        cv2.resize(vis, (720, int(H*sc))))
        return (m["image_id"], f"{W}x{H}", time.time() - t0)
    except Exception as e:
        return (m["image_id"], f"ERROR {type(e).__name__}: {e}", time.time() - t0)


if __name__ == "__main__":
    # rawpy + OpenMP deadlocks under fork; use forkserver.
    try:
        mp.set_start_method("forkserver")
    except RuntimeError:
        pass
    # process QA-subset first so overlays appear early, then the rest
    todo = [m for m in manifest if m["image_id"] in qa_ids] + \
           [m for m in manifest if m["image_id"] not in qa_ids]
    print(f"total {len(todo)} frames, {len(qa_ids)} QA overlays, {NWORKERS} workers", flush=True)
    t0 = time.time(); done = 0; errs = 0; times = []
    with mp.Pool(NWORKERS) as pool:
        for img_id, status, dt in pool.imap_unordered(process, todo):
            done += 1
            if status.startswith("ERROR"):
                errs += 1; print(f"  [{done}/{len(todo)}] {img_id} {status}", flush=True)
            elif status != "cached":
                times.append(dt)
            if done % 25 == 0 or done == len(todo):
                avg = sum(times)/len(times) if times else 0
                print(f"  [{done}/{len(todo)}] elapsed {time.time()-t0:.0f}s avg {avg:.1f}s/img errs {errs}", flush=True)
    print(f"DONE {done} frames, {errs} errors, wall {time.time()-t0:.0f}s", flush=True)
