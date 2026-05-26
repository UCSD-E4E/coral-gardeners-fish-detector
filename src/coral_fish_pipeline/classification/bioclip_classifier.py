from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import json
import shutil

import numpy as np
from PIL import Image

from coral_fish_pipeline.models import CropRecord, ClassificationResult
from coral_fish_pipeline.classification.species_prompts import build_species_prompt_map
from coral_fish_pipeline.utils.paths import safe_name

ALLOWED_BIOCLIP_MODELS = {
    "hf-hub:imageomics/bioclip-2.5-vith14",
    "hf-hub:imageomics/bioclip-2",
}
FORBIDDEN_MODELS = {"hf-hub:imageomics/bioclip", "imageomics/bioclip"}


class BioCLIPClassifier:
    def __init__(
        self,
        species: list[str],
        region: str,
        primary_model_id: str = "hf-hub:imageomics/bioclip-2.5-vith14",
        fallback_model_id: str = "hf-hub:imageomics/bioclip-2",
        device: str = "auto",
        precision: str = "fp16",
        batch_size: int = 1,
        unknown_threshold: float = 0.25,
        uncertain_margin: float = 0.08,
        cache_dir: str | Path | None = None,
        cache_text_embeddings: bool = True,
        classify_masked_crops: bool = True,
        allow_original_bioclip: bool = False,
    ) -> None:
        self.species = species
        self.region = region
        self.allow_original_bioclip = bool(allow_original_bioclip)
        self.primary_model_id = self._validate_model(primary_model_id)
        self.fallback_model_id = self._validate_model(fallback_model_id)
        self.device_request = device
        self.precision = precision
        self.batch_size = max(1, int(batch_size))
        self.unknown_threshold = float(unknown_threshold)
        self.uncertain_margin = float(uncertain_margin)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_text_embeddings = bool(cache_text_embeddings)
        self.classify_masked_crops = bool(classify_masked_crops)
        self.model_id: str | None = None
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.text_features = None
        self.device = None

    def _is_original_bioclip(self, model_id: str) -> bool:
        normalized = model_id.removeprefix("hf-hub:")
        for forbidden in FORBIDDEN_MODELS:
            marker = forbidden.removeprefix("hf-hub:")
            if normalized == marker:
                return True
            idx = normalized.find(marker)
            if idx >= 0:
                end = idx + len(marker)
                if end == len(normalized) or normalized[end] in {"/", ":", "@", "?"}:
                    return True
        return False

    def _validate_model(self, model_id: str) -> str:
        if self._is_original_bioclip(model_id):
            if not self.allow_original_bioclip:
                raise ValueError("Original BioCLIP is forbidden. Use BioCLIP 2.5 or BioCLIP 2 only.")
            return model_id
        if model_id not in ALLOWED_BIOCLIP_MODELS:
            raise ValueError(f"Unsupported BioCLIP model {model_id!r}. Allowed: {sorted(ALLOWED_BIOCLIP_MODELS)}")
        return model_id

    def _select_device(self):
        import torch
        if self.device_request == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device_request)

    def load(self) -> None:
        errors: list[str] = []
        for model_id in [self.primary_model_id, self.fallback_model_id]:
            try:
                self._load_model(model_id)
                self.model_id = model_id
                self.text_features = self._get_text_features()
                return
            except RuntimeError as exc:
                errors.append(f"{model_id}: {exc}")
                self._cleanup_cuda()
                continue
            except Exception as exc:
                errors.append(f"{model_id}: {exc}")
                self._cleanup_cuda()
                continue
        raise RuntimeError("Could not load BioCLIP 2.5 or BioCLIP 2. Errors:\n" + "\n".join(errors))

    def _load_model(self, model_id: str) -> None:
        import torch
        import open_clip

        self.device = self._select_device()
        self.model, _, preprocess_val = open_clip.create_model_and_transforms(model_id)
        self.preprocess = preprocess_val
        self.tokenizer = open_clip.get_tokenizer(model_id)
        self.model.eval()
        self.model.to(self.device)
        if self.precision == "fp16" and self.device.type == "cuda":
            self.model.half()
        # Tiny allocation + no-op encode path catches many OOM/load issues early.
        torch.cuda.empty_cache() if self.device.type == "cuda" else None

    def _cleanup_cuda(self) -> None:
        try:
            import torch
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _cache_path(self) -> Path | None:
        if not self.cache_dir or not self.model_id:
            return None
        payload = json.dumps({"model": self.model_id, "region": self.region, "species": self.species}, sort_keys=True)
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"text_embeddings_{safe_name(self.region)}_{digest}.pt"

    def _get_text_features(self):
        import torch
        import torch.nn.functional as F

        cache_path = self._cache_path()
        if self.cache_text_embeddings and cache_path and cache_path.exists():
            return torch.load(cache_path, map_location=self.device)

        prompt_map = build_species_prompt_map(self.species)
        features = []
        assert self.model is not None and self.tokenizer is not None and self.device is not None
        with torch.no_grad():
            for species in self.species:
                prompts = prompt_map[species]
                tokens = self.tokenizer(prompts).to(self.device)
                feats = self.model.encode_text(tokens)
                feats = F.normalize(feats.float(), dim=-1)
                mean_feat = F.normalize(feats.mean(dim=0, keepdim=True), dim=-1)
                features.append(mean_feat)
        text_features = torch.cat(features, dim=0)
        if self.cache_text_embeddings and cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(text_features.cpu(), cache_path)
            text_features = text_features.to(self.device)
        return text_features

    def _get_logit_scale(self):
        import torch

        assert self.device is not None
        if self.model is not None and hasattr(self.model, "logit_scale"):
            try:
                logit_scale = self.model.logit_scale
                if not isinstance(logit_scale, torch.Tensor):
                    logit_scale = torch.as_tensor(logit_scale, device=self.device)
                return logit_scale.detach().float().to(self.device).exp()
            except Exception:
                pass
        return torch.tensor(100.0, device=self.device)

    def _predict_image(self, image_path: str | Path) -> dict[str, Any]:
        import torch
        import torch.nn.functional as F

        if self.model is None or self.preprocess is None or self.text_features is None:
            self.load()
        assert self.model is not None and self.preprocess is not None and self.device is not None
        image = Image.open(image_path).convert("RGB")
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        if self.precision == "fp16" and self.device.type == "cuda":
            tensor = tensor.half()
        with torch.no_grad():
            try:
                use_amp = self.precision in {"fp16", "bf16"} and self.device.type == "cuda"
                dtype = torch.float16 if self.precision == "fp16" else torch.bfloat16
                with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_amp):
                    image_feat = self.model.encode_image(tensor)
                image_feat = F.normalize(image_feat.float(), dim=-1)
                text_features = F.normalize(self.text_features.to(self.device).float(), dim=-1)
                logit_scale = self._get_logit_scale()
                logits = logit_scale * (image_feat @ text_features.T)
                probs = logits.softmax(dim=-1).squeeze(0)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and self.model_id == self.primary_model_id:
                    self._fallback_after_oom()
                    return self._predict_image(image_path)
                raise
        topk = min(5, len(self.species))
        vals, idxs = probs.topk(topk)
        top1 = float(vals[0].detach().cpu()) if topk else 0.0
        top2 = float(vals[1].detach().cpu()) if topk > 1 else 0.0
        return {
            "species": self.species[int(idxs[0])],
            "confidence": top1,
            "top5": [(self.species[int(i)], float(v)) for v, i in zip(vals.detach().cpu(), idxs.detach().cpu())],
            "top1_probability": top1,
            "top2_probability": top2,
            "margin": top1 - top2,
        }

    def _fallback_after_oom(self) -> None:
        if self.primary_model_id == self.fallback_model_id:
            raise RuntimeError("CUDA out of memory on BioCLIP model and no distinct fallback is configured.")
        print("CUDA OOM with BioCLIP 2.5. Falling back to BioCLIP 2...")
        self._cleanup_cuda()
        self._load_model(self.fallback_model_id)
        self.model_id = self.fallback_model_id
        self.text_features = self._get_text_features()

    def classify_crop(self, crop: CropRecord) -> ClassificationResult:
        raw_pred = self._predict_image(crop.raw_crop_path)
        masked_pred = None
        final_pred = raw_pred
        if self.classify_masked_crops and crop.masked_crop_path and Path(crop.masked_crop_path).exists():
            masked_pred = self._predict_image(crop.masked_crop_path)
            if masked_pred["confidence"] > raw_pred["confidence"]:
                final_pred = masked_pred

        top5 = final_pred["top5"]
        top1 = float(top5[0][1]) if top5 else 0.0
        top2 = float(top5[1][1]) if len(top5) > 1 else 0.0
        if top1 < self.unknown_threshold:
            status = "unknown"
            predicted = "Unknown"
        elif top1 - top2 < self.uncertain_margin:
            status = "uncertain"
            predicted = final_pred["species"]
        else:
            status = "confident"
            predicted = final_pred["species"]
        return ClassificationResult(
            crop_id=crop.crop_id,
            predicted_species=predicted,
            confidence=top1,
            top5=top5,
            region=self.region,
            status=status,
            model_id=self.model_id or self.primary_model_id,
            raw_prediction=raw_pred,
            masked_prediction=masked_pred,
        )

    def classify_crops(self, crops: list[CropRecord]) -> list[ClassificationResult]:
        if self.model is None:
            self.load()
        results: list[ClassificationResult] = []
        for crop in crops:
            results.append(self.classify_crop(crop))
        return results


def sort_crops_by_species(crops: list[CropRecord], results: list[ClassificationResult], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    by_crop = {c.crop_id: c for c in crops}
    for result in results:
        crop = by_crop.get(result.crop_id)
        if crop is None:
            continue
        folder_name = "Unknown" if result.status == "unknown" else "Uncertain" if result.status == "uncertain" else safe_name(result.predicted_species)
        dst_dir = output_dir / folder_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        src = Path(crop.raw_crop_path)
        if src.exists():
            shutil.copyfile(src, dst_dir / src.name)
