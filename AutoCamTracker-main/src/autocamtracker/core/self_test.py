"""Self-test for the AutoCamTracker V1 runtime.

Run from VSCode with the "AutoCamTracker V1 Self Test" launch config, or:

    .venv/bin/python code/V1/self_test.py
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import traceback


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_ROOT / "code" / "model" / "yolo26s.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TEST_VIDEO = OUTPUT_DIR / "self_test_input.mp4"


def main() -> int:
    results: list[tuple[str, str, str]] = []

    results.append(run_check("dependencies", check_dependencies))
    results.append(run_check("identity_store", check_identity_store))
    results.append(run_check("identity_manager", check_identity_manager))
    results.append(run_check("feature_gallery", check_feature_gallery))
    results.append(run_check("auto_feature_sampler", check_auto_feature_sampler))
    results.append(run_check("model_load", check_model_load))
    results.append(run_check("video_input_pipeline", check_video_input_pipeline))
    results.append(run_check("webcam_probe", check_webcam_probe, warning_ok=True))

    print("\nAutoCamTracker V1 self-test summary")
    print("=" * 40)
    failed = False
    for name, status, detail in results:
        print(f"{status:>6}  {name}")
        if detail:
            print(f"        {detail}")
        if status == "FAIL":
            failed = True

    return 1 if failed else 0


def run_check(name, func, warning_ok: bool = False) -> tuple[str, str, str]:
    try:
        detail = func()
        return (name, "PASS", detail or "")
    except CameraPermissionBlocked as exc:
        return (name, "WARN" if warning_ok else "FAIL", str(exc))
    except Exception as exc:  # pragma: no cover - command-line diagnostic
        traceback.print_exc()
        return (name, "FAIL", str(exc))


def check_dependencies() -> str:
    import cv2
    import filterpy
    import mss
    import PIL
    import ultralytics

    return (
        f"python={sys.executable}; "
        f"ultralytics={ultralytics.__version__}; "
        f"cv2={cv2.__version__}; "
        f"filterpy={filterpy.__version__}; "
        f"mss={mss.__version__}; "
        f"Pillow={PIL.__version__}"
    )


def check_model_load() -> str:
    sys.path.insert(0, str(PROJECT_ROOT / "code" / "V1"))
    from video_detector import InputConfig, VideoDetector

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")

    detector = VideoDetector(
        InputConfig(
            model_path="yolo26s.pt",
            tracker_name="deepocsort",
            source_type="video_file",
        )
    )
    detector.load_model()
    return f"loaded {detector.config.model_path}"


def check_identity_store() -> str:
    sys.path.insert(0, str(PROJECT_ROOT / "code" / "V1"))
    from vehicle_identity_store import VehicleIdentityStore
    from video_detector import TrackedDetection

    detection = TrackedDetection(
        track_id=12,
        bbox=(10.0, 20.0, 90.0, 80.0),
        class_id=2,
        class_name="car",
        confidence=0.88,
        center=(50.0, 50.0),
        frame_index=1,
        timestamp=1.0,
        tracker_name="botsort",
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        store = VehicleIdentityStore(Path(temp_dir) / "identity.sqlite3")
        vehicle_id = store.create_vehicle(detection)
        stored = store.get_vehicle(vehicle_id)
        summary = store.summary()
        store.close()

    if stored is None or summary.vehicle_count != 1:
        raise RuntimeError("identity store failed to persist metadata")
    return f"vehicle_id={vehicle_id}; bbox={stored.bbox}; master={summary.master_feature_count}"


def check_feature_gallery() -> str:
    import cv2
    import numpy as np

    sys.path.insert(0, str(PROJECT_ROOT / "code" / "V1"))
    from feature_gallery import FeatureGallery
    from vehicle_identity_store import VehicleIdentityStore
    from video_detector import TrackedDetection

    frame = np.zeros((180, 260, 3), dtype=np.uint8)
    rng = np.random.default_rng(7)
    frame[40:130, 60:170] = rng.integers(40, 220, size=(90, 110, 3), dtype=np.uint8)
    cv2.rectangle(frame, (60, 40), (170, 130), (255, 255, 255), 2)
    detection = TrackedDetection(
        track_id=7,
        bbox=(60.0, 40.0, 170.0, 130.0),
        class_id=2,
        class_name="car",
        confidence=0.91,
        center=(115.0, 85.0),
        frame_index=3,
        timestamp=2.0,
        tracker_name="botsort",
    )

    class DummyExtractor:
        def extract(self, _frame, bbox):
            x1, y1, x2, y2 = bbox
            vector = np.array([x1 + x2, y1 + y2, x2 - x1, y2 - y1], dtype=np.float32)
            vector /= max(1e-6, float(np.linalg.norm(vector)))
            return vector.tolist()

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "identity.sqlite3"
        store = VehicleIdentityStore(db_path)
        vehicle_id = store.create_vehicle(detection)
        gallery = FeatureGallery(db_path)
        gallery.embedding_extractor = DummyExtractor()
        added = gallery.add_master_feature(vehicle_id, detection, frame)
        duplicate = gallery.add_master_feature(vehicle_id, detection, frame)
        matches = gallery.match_top_k(DummyExtractor().extract(frame, detection.bbox), vehicle_id=vehicle_id)
        counts = gallery.summary_by_vehicle()
        gallery.close()
        store.close()

    if not added.accepted or added.feature_id is None:
        raise RuntimeError(f"feature add failed: {added.reason}")
    if duplicate.accepted:
        raise RuntimeError("duplicate master feature was not rejected")
    if not matches or matches[0].vehicle_id != vehicle_id:
        raise RuntimeError("master gallery top-k matching failed")
    return f"vehicle_id={vehicle_id}; master={counts[vehicle_id].get('master', 0)}; top={matches[0].score:.2f}"


def check_identity_manager() -> str:
    import numpy as np

    sys.path.insert(0, str(PROJECT_ROOT / "code" / "V1"))
    from identity_manager import GlobalIdentityManager
    from vehicle_identity_store import VehicleIdentityStore
    from video_detector import TrackedDetection

    frame = np.full((120, 180, 3), 80, dtype=np.uint8)
    detection = TrackedDetection(
        track_id=31,
        bbox=(20.0, 20.0, 90.0, 80.0),
        class_id=2,
        class_name="car",
        confidence=0.74,
        center=(55.0, 50.0),
        frame_index=5,
        timestamp=1.0,
        tracker_name="botsort",
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        store = VehicleIdentityStore(Path(temp_dir) / "identity.sqlite3")
        manager = GlobalIdentityManager(identity_store=store)
        transient = manager.select_detection(detection, frame, persist=False)
        if transient.global_vehicle_id is not None or store.summary().vehicle_count != 0:
            raise RuntimeError("transient Auto Track selection wrote to Identity DB")

        vehicle_id = store.create_vehicle(detection)
        identity, score = manager.select_stored_vehicle(vehicle_id, [detection], frame)
        store.close()

    if identity is None or identity.last_track_id is not None or manager.status != "searching":
        raise RuntimeError("GID selection used local track fallback without master feature")
    return f"transient_gid={transient.global_vehicle_id}; masterless_score={score:.2f}"


def check_auto_feature_sampler() -> str:
    import cv2
    import numpy as np

    sys.path.insert(0, str(PROJECT_ROOT / "code" / "V1"))
    from auto_feature_sampler import AutoFeatureSampler
    from detection_store import DetectionStore
    from feature_gallery import FeatureGallery
    from vehicle_identity_store import VehicleIdentityStore
    from video_detector import TrackedDetection

    frame = np.zeros((220, 320, 3), dtype=np.uint8)
    rng = np.random.default_rng(11)
    frame[50:160, 80:220] = rng.integers(45, 230, size=(110, 140, 3), dtype=np.uint8)
    cv2.rectangle(frame, (80, 50), (220, 160), (255, 255, 255), 2)
    detection = TrackedDetection(
        track_id=41,
        bbox=(80.0, 50.0, 220.0, 160.0),
        class_id=2,
        class_name="car",
        confidence=0.92,
        center=(150.0, 105.0),
        frame_index=10,
        timestamp=3.0,
        tracker_name="botsort",
    )

    class DummyExtractor:
        def extract(self, _frame, bbox):
            x1, y1, x2, y2 = bbox
            vector = np.array([x1 + x2, y1 + y2, x2 - x1, y2 - y1], dtype=np.float32)
            vector /= max(1e-6, float(np.linalg.norm(vector)))
            return vector.tolist()

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "identity.sqlite3"
        store = VehicleIdentityStore(db_path)
        vehicle_id = store.create_vehicle(detection)
        gallery = FeatureGallery(db_path)
        gallery.embedding_extractor = DummyExtractor()
        detection_store = DetectionStore()
        detection_store.update([detection], frame.shape)
        sampler = AutoFeatureSampler(gallery)
        strict_area = None
        sampler.set_mode("Strict")
        strict_area = sampler.config.min_area_ratio
        sampler.set_mode("Diverse")
        diverse_area = sampler.config.min_area_ratio
        sampler.set_mode("Balanced")
        result = sampler.start(vehicle_id, detection, frame, detection_store)
        truck_detection = TrackedDetection(
            track_id=41,
            bbox=(80.0, 50.0, 220.0, 160.0),
            class_id=7,
            class_name="truck",
            confidence=0.92,
            center=(150.0, 105.0),
            frame_index=30,
            timestamp=4.0,
            tracker_name="botsort",
        )
        class_reject = sampler.sample(vehicle_id, truck_detection, frame, detection_store, force=True)
        counts = gallery.summary_by_vehicle()
        gallery.close()
        store.close()

    if strict_area is None or not diverse_area < strict_area:
        raise RuntimeError("auto feature modes did not loosen Diverse area gating")
    if not result.accepted or result.feature_id is None:
        raise RuntimeError(f"auto feature sampler failed: {result.reason}")
    if class_reject.accepted:
        raise RuntimeError("auto feature sampler accepted a mismatched class into Master")
    if counts[vehicle_id].get("master", 0) != 1:
        raise RuntimeError("auto feature sampler did not write a master feature")
    return f"vehicle_id={vehicle_id}; feature={result.feature_id}; quality={result.quality_score:.2f}"


def check_video_input_pipeline() -> str:
    import cv2
    import numpy as np

    sys.path.insert(0, str(PROJECT_ROOT / "code" / "V1"))
    from video_detector import InputConfig, VideoDetector

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(TEST_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        60.0,
        (320, 240),
    )
    if not writer.isOpened():
        raise RuntimeError("Unable to create self-test video file")

    for index in range(8):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        x = 30 + index * 8
        cv2.rectangle(frame, (x, 80), (x + 70, 150), (0, 255, 0), -1)
        cv2.putText(frame, "AutoCamTracker", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        writer.write(frame)
    writer.release()

    detector = VideoDetector(
        InputConfig(
            source_type="video_file",
            video_path=str(TEST_VIDEO),
            model_path="yolo26s.pt",
            tracker_name="deepocsort",
        )
    )
    detector.load_model()
    detector.open_source()
    source_fps = detector.get_source_fps()
    if source_fps is None or abs(source_fps - 60.0) > 1.0:
        detector.close()
        raise RuntimeError(f"Expected a 60fps source, got {source_fps}")
    frame, detections = detector.read_and_track()
    skipped = detector.skip_video_frames(2)
    detector.close()
    if frame is None:
        raise RuntimeError("Video input opened but returned no frame")
    return f"source_fps={source_fps:.1f}; read frame {frame.shape}; detections={len(detections)}; skipped={skipped}"


def check_webcam_probe() -> str:
    import cv2

    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    capture = cv2.VideoCapture(0, backend)
    opened = capture.isOpened()
    ok, frame = capture.read() if opened else (False, None)
    capture.release()

    if opened and ok and frame is not None:
        return f"camera index 0 ok; frame={frame.shape}"

    if sys.platform == "darwin":
        raise CameraPermissionBlocked(
            "Camera is not available to this Python process. "
            "Enable Camera permission for Visual Studio Code or Terminal in "
            "System Settings > Privacy & Security > Camera, then restart VSCode."
        )
    raise RuntimeError("Camera index 0 did not open")


class CameraPermissionBlocked(RuntimeError):
    pass


if __name__ == "__main__":
    raise SystemExit(main())
