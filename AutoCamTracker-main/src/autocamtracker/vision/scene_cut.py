"""Lightweight camera-cut detection for V1 video streams."""

from __future__ import annotations


class SceneCutDetector:
    """Detects hard cuts using a downscaled HSV histogram correlation."""

    def __init__(self, threshold: float = 0.62, confirm_frames: int = 2, cooldown_frames: int = 8) -> None:
        self.threshold = threshold
        self.confirm_frames = max(1, int(confirm_frames))
        self.cooldown_frames = max(0, int(cooldown_frames))
        self.previous_hist = None
        self._pending_cut_frames = 0
        self._cooldown_remaining = 0

    def reset(self) -> None:
        self.previous_hist = None
        self._pending_cut_frames = 0
        self._cooldown_remaining = 0

    def update(self, frame) -> bool:
        import cv2

        hist = self._histogram(frame)
        if self.previous_hist is None:
            self.previous_hist = hist
            return False

        correlation = cv2.compareHist(self.previous_hist, hist, cv2.HISTCMP_CORREL)
        self.previous_hist = hist
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            self._pending_cut_frames = 0
            return False

        if correlation >= self.threshold:
            self._pending_cut_frames = 0
            return False

        self._pending_cut_frames += 1
        if self._pending_cut_frames < self.confirm_frames:
            return False

        self._pending_cut_frames = 0
        self._cooldown_remaining = self.cooldown_frames
        return True

    @staticmethod
    def _histogram(frame):
        import cv2

        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist.astype("float32")
