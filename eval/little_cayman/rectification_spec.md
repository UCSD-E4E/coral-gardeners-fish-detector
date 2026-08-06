# FishSense-Lite ORF → Rectified reproduction spec

Source: fishsense-core==2.4.1 (external wheel, plain-Python image modules).

## Load raw (RawImage._get_data)
```python
import io, math, cv2, numpy as np, rawpy
from skimage.exposure import adjust_gamma, equalize_adapthist
from skimage.util import img_as_float, img_as_ubyte

def load_raw_bgr(orf_bytes):
    with rawpy.imread(io.BytesIO(orf_bytes)) as raw:
        img = img_as_float(raw.postprocess(
            gamma=(1,1), no_auto_bright=True, use_camera_wb=True,
            output_bps=16, user_flip=0))          # RGB float, no rotation
    hsv = cv2.cvtColor(img_as_ubyte(img), cv2.COLOR_BGR2HSV)  # quirk: BGR conv on RGB, scalar only
    mean = np.mean(cv2.split(hsv)[2])
    gamma = 1.0 / (math.log(20*255) / math.log(mean))
    img = adjust_gamma(img, gamma=gamma)
    img = equalize_adapthist(img)                  # CLAHE, skimage defaults
    return img_as_ubyte(img[:, :, ::-1])           # BGR uint8, full res

def rectify(bgr, K, D):   # K 3x3 row-major [[fx,0,cx],[0,fy,cy],[0,0,1]]; D [k1,k2,p1,p2,k3]
    return cv2.undistort(bgr, np.asarray(K,float), np.asarray(D,float).squeeze())
```

## Labels = identity in full-res rectified pixels
- headtaillabel.head_x/head_y/tail_x/tail_y and laserlabel.x/y are already full-res rectified pixels.
- Index directly: rectified[int(round(y)), int(round(x))]. No scale, no offset, no rotation.
- Snout=head, Fork=tail.

## Species labeling = whole frame (no crop). I build fish crop from head/tail myself.

## Versions for bit-parity: rawpy 0.27, scikit-image 0.26, opencv 4.13, numpy ~2.5.
## Rectified W×H not hardcoded = rawpy.postprocess output for the Olympus body (~5184×3888, confirm empirically). user_flip=0 => do NOT auto-rotate.
