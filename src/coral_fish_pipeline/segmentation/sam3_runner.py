from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image

from coral_fish_pipeline.models import Detection
from coral_fish_pipeline.utils.boxes import clip_box_xyxy, non_max_suppression
from coral_fish_pipeline.utils.masks import save_mask, mask_to_box


@dataclass
class _Candidate:
    prompt: str
    score: float
    bbox_xyxy: list[float]
    mask: np.ndarray | None


def iter_tiles(width: int, height: int, tile_size: int, tile_overlap: float) -> list[tuple[int, int, int, int]]:
    tile_size = max(1, int(tile_size))
    overlap = min(max(float(tile_overlap), 0.0), 0.95)
    stride = max(1, int(round(tile_size * (1.0 - overlap))))

    xs = [0]
    ys = [0]
    while xs[-1] + tile_size < width:
        xs.append(min(xs[-1] + stride, max(0, width - tile_size)))
        if xs[-1] == max(0, width - tile_size):
            break
    while ys[-1] + tile_size < height:
        ys.append(min(ys[-1] + stride, max(0, height - tile_size)))
        if ys[-1] == max(0, height - tile_size):
            break

    boxes: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for y in ys:
        for x in xs:
            x1 = max(0, min(int(x), max(0, width - 1)))
            y1 = max(0, min(int(y), max(0, height - 1)))
            x2 = min(width, x1 + tile_size)
            y2 = min(height, y1 + tile_size)
            if x2 <= x1 or y2 <= y1:
                continue
            tile = (x1, y1, x2, y2)
            if tile not in seen:
                boxes.append(tile)
                seen.add(tile)
    return boxes


def offset_box_xyxy(box: list[float], offset_x: int, offset_y: int, width: int, height: int) -> list[float]:
    return clip_box_xyxy([box[0] + offset_x, box[1] + offset_y, box[2] + offset_x, box[3] + offset_y], width, height)


class SAM3Runner:
    """
    Real SAM3 adapter.

    Uses official SAM3 API:
      build_sam3_image_model()
      Sam3Processor(model)
      processor.set_image(...)
      processor.set_text_prompt(...)
    """

    def __init__(
        self,
        min_confidence: float = 0.25,
        max_detections_per_image: int = 100,
        nms_iou_threshold: float = 0.85,
        use_tiling: bool = False,
        tile_size: int = 768,
        tile_overlap: float = 0.25,
        **kwargs: object,
    ) -> None:
        self.min_confidence = float(min_confidence)
        self.max_detections = int(max_detections_per_image)
        self.nms_iou_threshold = float(nms_iou_threshold)
        self.use_tiling = bool(use_tiling)
        self.tile_size = int(tile_size)
        self.tile_overlap = float(tile_overlap)
        self.kwargs = kwargs

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.autocast_dtype: torch.dtype | None = None
        if self.device == "cuda":
            self.autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except Exception as exc:
            raise RuntimeError(
                "SAM3 backend selected, but SAM3 could not be imported. "
                "Make sure you installed facebookresearch/sam3 in this WSL environment."
            ) from exc

        self.model = build_sam3_image_model()
        self.model.to(self.device)
        self.model.eval()
        self.processor = Sam3Processor(self.model)
        autocast_name = "none" if self.autocast_dtype is None else str(self.autocast_dtype).removeprefix("torch.")
        print(f"SAM3 backend loaded on {self.device} with autocast={autocast_name}")

    def _autocast_context(self):
        if self.device == "cuda" and self.autocast_dtype is not None:
            return torch.autocast(device_type="cuda", dtype=self.autocast_dtype)
        return nullcontext()

    def _to_numpy(self, value):
        if value is None:
            return np.array([])

        if isinstance(value, torch.Tensor):
            value = value.detach()

            if value.dtype in (torch.bfloat16, torch.float16):
                value = value.to(torch.float32)

            return value.cpu().numpy()

        return np.asarray(value)

    def _normalize_mask(self, mask: Any, image_size: tuple[int, int]) -> np.ndarray:
        arr = self._to_numpy(mask)

        # Common shapes: [H, W], [1, H, W], [H, W, 1]
        arr = np.squeeze(arr)

        if arr.ndim != 2:
            raise ValueError(f"Unexpected SAM3 mask shape: {arr.shape}")

        mask_bool = arr > 0

        # If mask is not image-sized for any reason, resize nearest-neighbor.
        w, h = image_size
        if mask_bool.shape != (h, w):
            mask_img = Image.fromarray(mask_bool.astype(np.uint8) * 255, mode="L")
            mask_img = mask_img.resize((w, h), resample=Image.Resampling.NEAREST)
            mask_bool = np.asarray(mask_img) > 0

        return mask_bool

    def _normalize_box(self, box: Any) -> list[float] | None:
        arr = self._to_numpy(box).reshape(-1)
        if arr.size < 4:
            return None
        x1, y1, x2, y2 = arr[:4].astype(float).tolist()
        return [x1, y1, x2, y2]

    def _box_for_index(self, boxes: Any, index: int) -> list[float] | None:
        arr = self._to_numpy(boxes)
        if arr.size == 0:
            return None
        if arr.ndim == 1:
            return self._normalize_box(arr) if index == 0 else None
        if index >= len(arr):
            return None
        return self._normalize_box(arr[index])

    def _output_value(self, output: Any, *keys: str) -> tuple[Any, str | None]:
        for key in keys:
            if isinstance(output, Mapping) and key in output:
                return output[key], key
            if hasattr(output, key):
                return getattr(output, key), key
        return None, None

    def _extract_outputs(self, output: Any) -> tuple[Any, Any, Any, str | None]:
        masks, _ = self._output_value(output, "masks", "pred_masks")
        boxes, _ = self._output_value(output, "boxes", "pred_boxes")
        scores, score_key = self._output_value(output, "scores", "logits", "iou_scores")
        return masks, boxes, scores, score_key

    def _iter_masks(self, masks: Any) -> list[np.ndarray]:
        arr = self._to_numpy(masks)
        if arr.size == 0:
            return []
        if arr.ndim == 2:
            return [arr]
        if arr.ndim == 3:
            return [arr[i] for i in range(arr.shape[0])]
        if arr.ndim == 4:
            if arr.shape[1] == 1:
                arr = arr[:, 0, :, :]
            elif arr.shape[-1] == 1:
                arr = arr[..., 0]
            else:
                arr = np.squeeze(arr)
                if arr.ndim == 2:
                    return [arr]
                if arr.ndim != 3:
                    raise ValueError(f"Unexpected SAM3 masks shape: {arr.shape}")
            return [arr[i] for i in range(arr.shape[0])]
        arr = np.squeeze(arr)
        if arr.ndim == 2:
            return [arr]
        if arr.ndim == 3:
            return [arr[i] for i in range(arr.shape[0])]
        raise ValueError(f"Unexpected SAM3 masks shape: {arr.shape}")

    def _normalize_scores(self, scores: Any, score_key: str | None) -> np.ndarray | None:
        if scores is None:
            return None
        arr = self._to_numpy(scores).astype(float).reshape(-1)
        if arr.size == 0:
            return None
        if score_key == "logits" and ((arr < 0.0).any() or (arr > 1.0).any()):
            arr = 1.0 / (1.0 + np.exp(-arr))
        return arr

    def _full_size_mask(
        self,
        tile_mask: np.ndarray,
        tile_box: tuple[int, int, int, int] | None,
        full_size: tuple[int, int],
    ) -> np.ndarray:
        full_w, full_h = full_size
        if tile_box is None:
            return tile_mask
        x1, y1, x2, y2 = tile_box
        full = np.zeros((full_h, full_w), dtype=bool)
        full[y1:y2, x1:x2] = tile_mask[: y2 - y1, : x2 - x1]
        return full

    def _predict_candidates(
        self,
        image: Image.Image,
        prompts: list[str],
        full_size: tuple[int, int],
        tile_box: tuple[int, int, int, int] | None = None,
    ) -> list[_Candidate]:
        image = image.convert("RGB")
        width, height = image.size
        full_width, full_height = full_size
        offset_x = tile_box[0] if tile_box else 0
        offset_y = tile_box[1] if tile_box else 0
        candidates: list[_Candidate] = []

        prompt_outputs: list[tuple[str, Any]] = []
        with torch.inference_mode(), self._autocast_context():
            state = self.processor.set_image(image)
            for prompt in prompts:
                prompt_outputs.append((prompt, self.processor.set_text_prompt(state=state, prompt=prompt)))

        for prompt, output in prompt_outputs:
            masks, boxes, scores, score_key = self._extract_outputs(output)

            boxes_np = None if boxes is None else self._to_numpy(boxes)
            scores_np = self._normalize_scores(scores, score_key)
            if masks is None:
                if boxes_np is None:
                    continue
                box_count = 1 if boxes_np.ndim == 1 else len(boxes_np)
                for i in range(box_count):
                    score = 1.0
                    if scores_np is not None and i < len(scores_np):
                        score = float(scores_np[i])
                    if score < self.min_confidence:
                        continue
                    bbox = self._box_for_index(boxes_np, i)
                    if bbox is None:
                        continue
                    bbox = clip_box_xyxy(bbox, width, height)
                    if tile_box is not None:
                        bbox = offset_box_xyxy(bbox, offset_x, offset_y, full_width, full_height)
                    candidates.append(_Candidate(prompt=prompt, score=score, bbox_xyxy=bbox, mask=None))
                continue

            mask_items = self._iter_masks(masks)

            for i, raw_mask in enumerate(mask_items):
                score = 1.0
                if scores_np is not None and i < len(scores_np):
                    score = float(scores_np[i])

                if score < self.min_confidence:
                    continue

                try:
                    mask_bool = self._normalize_mask(raw_mask, image.size)
                except Exception:
                    continue

                bbox = self._box_for_index(boxes_np, i) if boxes_np is not None else None
                if bbox is None:
                    bbox = mask_to_box(mask_bool)

                if bbox is None:
                    continue

                bbox = clip_box_xyxy(bbox, width, height)
                if tile_box is not None:
                    bbox = offset_box_xyxy(bbox, offset_x, offset_y, full_width, full_height)
                    mask_bool = self._full_size_mask(mask_bool, tile_box, full_size)
                candidates.append(_Candidate(prompt=prompt, score=score, bbox_xyxy=bbox, mask=mask_bool))

        return candidates

    def _dedupe_candidates(self, candidates: list[_Candidate]) -> list[_Candidate]:
        return non_max_suppression(candidates, self.nms_iou_threshold)[: self.max_detections]

    def predict(
        self,
        image: Image.Image,
        image_id: str,
        output_mask_dir: str | Path,
        prompts: list[str] | None = None,
    ) -> list[Detection]:
        output_mask_dir = Path(output_mask_dir)
        output_mask_dir.mkdir(parents=True, exist_ok=True)

        prompts = prompts or ["fish"]
        image = image.convert("RGB")
        width, height = image.size

        candidates = self._predict_candidates(image, prompts, full_size=image.size)
        if self.use_tiling:
            for tile_box in iter_tiles(width, height, self.tile_size, self.tile_overlap):
                tile = image.crop(tile_box)
                candidates.extend(self._predict_candidates(tile, prompts, full_size=image.size, tile_box=tile_box))

        detections: list[Detection] = []
        for idx, candidate in enumerate(self._dedupe_candidates(candidates)):
            det_id = f"{image_id}_{idx:04d}"
            mask_path: Path | None = None
            if candidate.mask is not None:
                mask_path = output_mask_dir / f"{det_id}.png"
                save_mask(candidate.mask, mask_path)

            detections.append(
                Detection(
                    image_id=image_id,
                    det_id=det_id,
                    bbox_xyxy=[float(v) for v in candidate.bbox_xyxy],
                    mask_path=str(mask_path) if mask_path is not None else None,
                    score=float(candidate.score),
                    prompt=candidate.prompt,
                    status="accepted",
                    rejection_reason=None,
                )
            )

        return detections
