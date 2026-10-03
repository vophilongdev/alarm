"""
stream_runner.py — Luồng đọc RTSP camera và chạy YOLO detect lấn chiếm tự động.
"""
import logging
import threading
import time
import cv2

from core.settings import settings
from services.alarm_client import create_alarm, patch_camera
from services.camera_registry import registry
from services.encroach_tracker import get_tracker, parse_roi
from services.yolo_model import detect_frame

logger = logging.getLogger(__name__)


def get_rtsp_url(cam: dict) -> str:
    storage_url = cam.get("storage_url")
    if storage_url and str(storage_url).strip() and str(storage_url).strip() != "null":
        username = cam.get("storage_username") or "admin"
        password = cam.get("storage_password") or ""
        channel = cam.get("storage_channel") or 1
        port = cam.get("storage_port") or 554
        try:
            port = int(port)
            if port not in (554, 5544, 8554):
                port = 554
        except Exception:
            port = 554
        return f"rtsp://{username}:{password}@{storage_url}:{port}/cam/realmonitor?channel={channel}&subtype=0"

    raw_url = cam.get("url") or ""
    if raw_url.startswith("rtsp://") or raw_url.startswith("http://") or raw_url.startswith("https://"):
        return raw_url

    return ""


def get_zone_from_param(param: dict) -> list:
    if not isinstance(param, dict):
        return []
    for k in ("encroach", "sidewalk_occupation", "region"):
        sub = param.get(k)
        if isinstance(sub, dict):
            zone = sub.get("zone") or sub.get("condition")
            if zone and isinstance(zone, list):
                return zone
    return param.get("zone") or []

class CameraWorkerThread(threading.Thread):
    def __init__(self, cam_id: str, servicer):
        super().__init__(name=f"cam-worker-{cam_id}", daemon=True)
        self.cam_id = cam_id
        self.servicer = servicer
        self.stopped = threading.Event()

    def stop(self):
        self.stopped.set()

    def run(self):
        logger.info("[%s] Bắt đầu luồng tự động đọc camera", self.cam_id)
        while not self.stopped.is_set():
            cam = registry.get(self.cam_id)
            if not cam:
                time.sleep(2)
                continue

            use_sdk = cam.get("use_sdk", 0)
            manufacturer = str(cam.get("manufacturer") or "dahua").lower()

            if use_sdk == 1:
                storage_url = cam.get("storage_url")
                storage_port = cam.get("storage_port") or 8000
                storage_username = cam.get("storage_username") or "admin"
                storage_password = cam.get("storage_password") or ""
                storage_channel = cam.get("storage_channel") or 1

                if not storage_url:
                    logger.warning("[%s] use_sdk=1 nhưng không có storage_url hợp lệ", self.cam_id)
                    time.sleep(5)
                    continue

                logger.info("[%s] 🎥 Bắt đầu luồng SDK (%s) cho camera: %s:%s (channel=%s)",
                            self.cam_id, manufacturer, storage_url, storage_port, storage_channel)

                viewer = None
                try:
                    if manufacturer in ("hik", "hikvision"):
                        from services.sdk.hik_sdk_realplay import HIKCameraViewer
                        viewer = HIKCameraViewer()
                        connected = viewer.connect(storage_url, int(storage_port), storage_username, storage_password)
                        if connected:
                            viewer.start_stream(int(storage_channel), 0)
                    else:
                        from services.sdk.dha_sdk_realplay import DahuaCameraViewer
                        viewer = DahuaCameraViewer()
                        connected = viewer.connect(storage_url, int(storage_port), storage_username, storage_password)
                        if connected:
                            ch = int(storage_channel) - 1 if int(storage_channel) > 0 else 0
                            viewer.start_stream(ch, 0)
                except Exception as e:
                    logger.error("[%s] Lỗi khởi tạo SDK %s: %s", self.cam_id, manufacturer, e)

                if not viewer or not getattr(viewer, "is_connected", False):
                    logger.warning("[%s] Thất bại kết nối SDK (%s). Thử lại sau 5s", self.cam_id, manufacturer)
                    if viewer:
                        try:
                            viewer.cleanup()
                        except Exception:
                            pass
                    time.sleep(5)
                    continue

                logger.info("[%s] ✅ Đã kết nối SDK camera thành công: %s:%s", self.cam_id, storage_url, storage_port)
                patch_camera(self.cam_id, 1)
                last_frame_time = 0.0

                while not self.stopped.is_set() and getattr(viewer, "is_playing", True):
                    frame = viewer.get_frame()
                    if frame is None:
                        time.sleep(0.05)
                        continue

                    now = time.monotonic()
                    if now - last_frame_time < 0.5:  # Giới hạn ~2 FPS cho GPU
                        continue
                    last_frame_time = now

                    orig_h, orig_w = frame.shape[:2]
                    img_resized = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_AREA)

                    try:
                        detections = detect_frame(self.servicer.model, self.servicer.device,
                                                  self.servicer.model_is_pt, img_resized,
                                                  orig_w, orig_h)
                    except Exception as e:
                        logger.error("[%s] Lỗi YOLO detect: %s", self.cam_id, e)
                        continue

                    cam_latest = registry.get(self.cam_id) or cam
                    zone = get_zone_from_param(cam_latest.get("parameter"))
                    roi_contour = parse_roi(zone, orig_w, orig_h)

                    formatted_dets = [{
                        "bbox": (b["x1"], b["y1"], b["x2"], b["y2"]),
                        "class_name": b["class_name"],
                        "confidence": b["confidence"],
                    } for b in detections]

                    tracker = get_tracker(self.cam_id)
                    results, alarm_ids = tracker.update(formatted_dets, roi_contour)

                    if alarm_ids:
                        logger.info("🚨 [%s] KÍCH HOẠT CẢNH BÁO LẤN CHIẾM (SDK)! (tracks=%s)", self.cam_id, alarm_ids)
                        evidence = self.servicer._draw_evidence(frame, results, roi_contour, orig_w, orig_h, alarm_ids)
                        alarm_set = set(alarm_ids)
                        tracks = [r for r in results if r["track_id"] in alarm_set]
                        self.servicer.alarm_pool.submit(self.servicer._post_alarm, self.cam_id, tracks, evidence)

                try:
                    viewer.cleanup()
                except Exception as e:
                    logger.error("[%s] Lỗi cleanup SDK: %s", self.cam_id, e)
                patch_camera(self.cam_id, 0)
                time.sleep(2)

            else:
                rtsp_url = get_rtsp_url(cam)
                if not rtsp_url:
                    logger.warning("[%s] Không tìm thấy RTSP URL hợp lệ", self.cam_id)
                    time.sleep(5)
                    continue

                cap = cv2.VideoCapture(rtsp_url)
                if not cap.isOpened():
                    logger.warning("[%s] Thất bại kết nối RTSP: %s. Thử lại sau 5s", self.cam_id, rtsp_url)
                    time.sleep(5)
                    continue

                logger.info("[%s] ✅ Đã kết nối RTSP camera thành công: %s", self.cam_id, rtsp_url)
                patch_camera(self.cam_id, 1)
                last_frame_time = 0.0

                while not self.stopped.is_set():
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        logger.warning("[%s] Mất tín hiệu RTSP frame, kết nối lại...", self.cam_id)
                        break

                    now = time.monotonic()
                    if now - last_frame_time < 0.5:  # Giới hạn ~2 FPS cho GPU
                        continue
                    last_frame_time = now

                    orig_h, orig_w = frame.shape[:2]
                    img_resized = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_AREA)

                    try:
                        detections = detect_frame(self.servicer.model, self.servicer.device,
                                                  self.servicer.model_is_pt, img_resized,
                                                  orig_w, orig_h)
                    except Exception as e:
                        logger.error("[%s] Lỗi YOLO detect: %s", self.cam_id, e)
                        continue

                    cam_latest = registry.get(self.cam_id) or cam
                    zone = get_zone_from_param(cam_latest.get("parameter"))
                    roi_contour = parse_roi(zone, orig_w, orig_h)

                    formatted_dets = [{
                        "bbox": (b["x1"], b["y1"], b["x2"], b["y2"]),
                        "class_name": b["class_name"],
                        "confidence": b["confidence"],
                    } for b in detections]

                    tracker = get_tracker(self.cam_id)
                    results, alarm_ids = tracker.update(formatted_dets, roi_contour)

                    if alarm_ids:
                        logger.info("🚨 [%s] KÍCH HOẠT CẢNH BÁO LẤN CHIẾM! (tracks=%s)", self.cam_id, alarm_ids)
                        evidence = self.servicer._draw_evidence(frame, results, roi_contour, orig_w, orig_h, alarm_ids)
                        alarm_set = set(alarm_ids)
                        tracks = [r for r in results if r["track_id"] in alarm_set]
                        self.servicer.alarm_pool.submit(self.servicer._post_alarm, self.cam_id, tracks, evidence)

                cap.release()
                patch_camera(self.cam_id, 0)
                time.sleep(2)


class StreamRunnerManager:
    def __init__(self, servicer):
        self.servicer = servicer
        self.workers = {}  # cam_id -> CameraWorkerThread
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._monitor_loop, name="stream-runner-mgr", daemon=True)
        self._thread.start()

    def _monitor_loop(self):
        while not self._stop.wait(5):
            current_cams = set(registry._cameras.keys())
            with self._lock:
                for cam_id in current_cams:
                    if cam_id not in self.workers or not self.workers[cam_id].is_alive():
                        w = CameraWorkerThread(cam_id, self.servicer)
                        self.workers[cam_id] = w
                        w.start()

                for cam_id in list(self.workers.keys()):
                    if cam_id not in current_cams:
                        self.workers[cam_id].stop()
                        del self.workers[cam_id]

    def stop(self):
        self._stop.set()
        with self._lock:
            for w in self.workers.values():
                w.stop()
            self.workers.clear()
