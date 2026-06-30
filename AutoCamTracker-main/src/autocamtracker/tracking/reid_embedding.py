"""Vehicle ReID embedding extraction for AutoCamTracker V1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = PROJECT_ROOT / "code" / "model"


@dataclass
class ReIDEmbeddingConfig:
    model_path: str = "yolo26s-reid.onnx"
    enabled: bool = True


class ReIDEmbeddingExtractor:
    """Thin wrapper around Ultralytics' ReID encoder."""

    def __init__(self, config: ReIDEmbeddingConfig | None = None) -> None:
        self.config = config or ReIDEmbeddingConfig()
        self.encoder: Any | None = None
        self.available = False
        self.error: str | None = None
        if self.config.enabled:
            self._load()

    def extract(self, frame, bbox: tuple[float, float, float, float]) -> list[float] | None:
        batch_features = self.extract_batch(frame, [bbox])
        return batch_features[0] if batch_features else None

    def extract_batch(self, frame, bboxes: list[tuple[float, float, float, float]]) -> list[list[float]] | None:
        if not self.available or self.encoder is None or not bboxes:
            return None
        import numpy as np

        dets_list = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            width = max(1.0, x2 - x1)
            height = max(1.0, y2 - y1)
            dets_list.append([(x1 + x2) / 2.0, (y1 + y2) / 2.0, width, height])

        dets = np.array(dets_list, dtype=np.float32)
        try:
            features = self.encoder(frame, dets)
        except Exception as exc:
            self.available = False
            self.error = str(exc)
            return None
            
        if not features:
            return None
            
        result = []
        for feature in features:
            feat = np.asarray(feature, dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(feat))
            if norm <= 1e-12:
                result.append([])
            else:
                result.append((feat / norm).astype("float32").tolist())
        return result

    def _load(self) -> None:
        try:
            from ultralytics.trackers.utils.reid import ReID

            self.encoder = ReID(str(self._resolve_model_path(self.config.model_path)))
            self.available = True
        except Exception as exc:
            self.encoder = None
            self.available = False
            self.error = str(exc)

    @staticmethod
    def _resolve_model_path(model_path: str) -> Path | str:
        path = Path(model_path).expanduser()
        if path.is_absolute():
            return path
        candidates = [MODEL_DIR / path, PROJECT_ROOT / path, Path.cwd() / path, path]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return model_path
