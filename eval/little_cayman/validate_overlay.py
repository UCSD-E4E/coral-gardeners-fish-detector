"""Render rectified frames with head/tail + laser overlays for visual QA."""
import json, os, sys, math
import cv2
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from rectify_lib import load_raw_bgr, rectify, load_intrinsics

SP = os.path.dirname(__file__)
manifest = json.load(open(os.path.join(SP, "manifest.json")))
intr = load_intrinsics(os.path.join(SP, "gt", "cam_intrinsics.tsv"))
OUT = os.path.join(SP, "qa")
os.makedirs(OUT, exist_ok=True)


def crop_box(ht, W, H, pad=0.4):
    xs = [ht["head_x"], ht["tail_x"]]
    ys = [ht["head_y"], ht["tail_y"]]
    length = math.hypot(ht["head_x"] - ht["tail_x"], ht["head_y"] - ht["tail_y"])
    m = pad * length
    x0 = max(0, int(min(xs) - m)); x1 = min(W, int(max(xs) + m))
    y0 = max(0, int(min(ys) - m)); y1 = min(H, int(max(ys) + m))
    return x0, y0, x1, y1


# pick up to 2 single-fish samples per species, prefer having a laser point
by_sp = {}
for m in manifest:
    if m["n_fish"] != 1:
        continue
    by_sp.setdefault(m["species"]["scientific"], []).append(m)

samples = []
for sp, items in by_sp.items():
    items.sort(key=lambda m: (0 if m["laser"] else 1))  # laser first
    samples.extend(items[:2])

print(f"rendering {len(samples)} samples", flush=True)
thumbs = []
for m in samples:
    thumb_path = os.path.join(OUT, f'thumb_{m["image_id"]}.jpg')
    if os.path.isfile(thumb_path):
        thumbs.append(cv2.imread(thumb_path))
        print(f'  {m["image_id"]:>6} (cached)', flush=True)
        continue
    K, D = intr[m["camera_id"]]
    bgr = load_raw_bgr(m["orf_path"])
    rect = rectify(bgr, K, D)
    H, W = rect.shape[:2]
    vis = rect.copy()
    ht = m["headtail"][0]
    hp = (int(ht["head_x"]), int(ht["head_y"]))
    tp = (int(ht["tail_x"]), int(ht["tail_y"]))
    cv2.line(vis, hp, tp, (0, 255, 255), 6)
    cv2.circle(vis, hp, 22, (0, 255, 0), -1)   # head green
    cv2.circle(vis, tp, 22, (0, 0, 255), -1)   # tail red
    for L in m["laser"]:
        cv2.circle(vis, (int(L["x"]), int(L["y"])), 18, (0, 165, 255), 3)  # laser orange ring
    x0, y0, x1, y1 = crop_box(ht, W, H)
    cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 0, 255), 4)  # crop magenta
    label = f'{m["species"]["common"]}  img{m["image_id"]}  {W}x{H}'
    cv2.putText(vis, label, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 6)
    cv2.putText(vis, label, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 2)
    # thumbnail for contact sheet
    scale = 640.0 / W
    th = cv2.resize(vis, (640, int(H * scale)))
    cv2.imwrite(thumb_path, th)
    thumbs.append(th)
    # also save the crop itself
    crop = rect[y0:y1, x0:x1]
    cv2.imwrite(os.path.join(OUT, f'crop_{m["species"]["scientific"].replace(" ","_")}_{m["image_id"]}.jpg'), crop)
    print(f'  {m["image_id"]:>6} {m["species"]["common"]:<22} {W}x{H} laser={len(m["laser"])}', flush=True)

# contact sheet: pad thumbs to same height, tile in grid
maxh = max(t.shape[0] for t in thumbs)
thumbs = [cv2.copyMakeBorder(t, 0, maxh - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(40,40,40)) for t in thumbs]
cols = 4
rows = []
for i in range(0, len(thumbs), cols):
    row = thumbs[i:i+cols]
    while len(row) < cols:
        row.append(np.full_like(thumbs[0], 40))
    rows.append(np.hstack(row))
sheet = np.vstack(rows)
cv2.imwrite(os.path.join(OUT, "contact_sheet.jpg"), sheet)
print("wrote", os.path.join(OUT, "contact_sheet.jpg"), sheet.shape)
