"""ORF -> rectified reproduction of fishsense-core 2.4.1, + label helpers.

Mirrors fishsense_core.image.raw_image.RawImage / rectified_image.RectifiedImage.
"""
import io, math, json, os
import cv2
import numpy as np
import rawpy
from skimage.exposure import adjust_gamma, equalize_adapthist
from skimage.util import img_as_float, img_as_ubyte


def load_raw_bgr(orf_path):
    with open(orf_path, "rb") as fh:
        data = fh.read()
    with rawpy.imread(io.BytesIO(data)) as raw:
        img = img_as_float(raw.postprocess(
            gamma=(1, 1), no_auto_bright=True, use_camera_wb=True,
            output_bps=16, user_flip=0))
    # auto-gamma from mean brightness (verbatim quirk: BGR2HSV on RGB, scalar use only)
    hsv = cv2.cvtColor(img_as_ubyte(img), cv2.COLOR_BGR2HSV)
    mean = float(np.mean(cv2.split(hsv)[2]))
    gamma = 1.0 / (math.log(20 * 255) / math.log(mean))
    img = adjust_gamma(img, gamma=gamma)
    img = equalize_adapthist(img)
    return img_as_ubyte(img[:, :, ::-1])  # BGR uint8, full res


def rectify(bgr, K, D):
    K = np.asarray(K, dtype=float)
    D = np.asarray(D, dtype=float).squeeze()
    return cv2.undistort(bgr, K, D)


def load_intrinsics(path):
    """cam_id -> (K, D) from staged TSV: cam \\t matrix_json \\t dist_json"""
    out = {}
    with open(path) as fh:
        for line in fh:
            cam, mtx, dist = line.rstrip("\n").split("\t")
            out[int(cam)] = (json.loads(mtx), json.loads(dist))
    return out
