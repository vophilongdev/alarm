"""
server.py — gRPC service phát hiện vũ khí (GRPC_DET_WEAPON).

Luồng một request:
    DetectFrame(image, stream_id)
      → kiểm tra stream_id có license_key + service_type hợp lệ
      → YOLO-seg detect (dynamic batching) → masks + bboxes
      → WeaponOverlapDetector: mask người & mask vũ khí có giao thoa?
      → Nếu giao thoa → CHỈ trả về vũ khí & POST /api/alarms (chụp tức thì)

Khác biệt chính so với grpc_det_encroach:
  - Segmentation thay vì Detection
  - Mask Overlap thay vì ROI + Dwell Time
  - Cảnh báo tức thì (zero dwell) thay vì đợi N giây
  - Chỉ trả vũ khí lên UI, class người là trigger nội bộ
"""
import logging
import os
import queue
import threading
import time
from collections import Counter
from concurrent import futures

import cv2
import grpc
import numpy as np
import torch

from core.settings import settings
from services import detector_pb2
from services import detector_pb2_grpc
from services.alarm_client import create_alarm
from services.camera_registry import registry
from services.weapon_overlap import WeaponOverlapDetector
from services.yolo_seg_model import init_yolo, detect_batch_seg

BATCH_SIZE = settings.BATCH_SIZE
BATCH_TIMEOUT_SEC = settings.BATCH_TIMEOUT_MS / 1000.0

logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s [{settings.LOG_TAG}] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


class YoloWeaponServiceServicer(detector_pb2_grpc.YoloWeaponServiceServicer):
    def __init__(self):
        self.model, self.device, self.model_is_pt = init_yolo()
        self.batch_queue = queue.Queue()
        self.shutdown_flag = threading.Event()
        self.alarm_pool = futures.ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="alarm")
        self.overlap_detector = WeaponOverlapDetector()

        self.worker_thread = threading.Thread(target=self._batch_worker_loop, daemon=True)
        self.worker_thread.start()
        logger.info("Dynamic batching bật (BATCH_SIZE=%d, TIMEOUT=%.0fms). "
                     "Model YOLO-seg sẵn sàng.",
                    BATCH_SIZE, BATCH_TIMEOUT_SEC * 1000)

    def _batch_worker_loop(self):
        while not self.shutdown_flag.is_set():
            batch_requests = []
            t_start_wait = time.time()

            while len(batch_requests) < BATCH_SIZE:
                remaining_time = BATCH_TIMEOUT_SEC - (time.time() - t_start_wait)
                if remaining_time <= 0 and batch_requests:
                    break
                try:
                    batch_requests.append(self.batch_queue.get(timeout=max(0.001, remaining_time)))
                except queue.Empty:
                    if batch_requests:
                        break
                    continue

            if not batch_requests:
                continue

            t_gpu_start = time.perf_counter()
            try:
                batch_results = detect_batch_seg(
                    model=self.model,
                    device=self.device,
                    model_is_pt=self.model_is_pt,
                    batch_items=[r["item"] for r in batch_requests],
                )
            except Exception as e:
                logger.error("Batch detect-seg lỗi: %s", e)
                if self.device != "cpu":
                    torch.cuda.empty_cache()
                batch_results = [[] for _ in batch_requests]

            t_gpu_dur_ms = (time.perf_counter() - t_gpu_start) * 1000.0

            total_detected = 0
            class_counts = Counter()
            for idx, req_tuple in enumerate(batch_requests):
                dets = batch_results[idx]
                req_tuple["result"] = dets
                for det in dets:
                    total_detected += 1
                    class_counts[det["class_name"]] += 1
                req_tuple["event"].set()

            class_summary = ", ".join(f"{k}: {v}" for k, v in class_counts.items()) or "none"
            logger.info("Batch: %d/%d | GPU: %.1fms | Detected: %d [%s]",
                        len(batch_requests), BATCH_SIZE, t_gpu_dur_ms,
                        total_detected, class_summary)

    def DetectFrame(self, request, context):
        stream_id = request.stream_id

        # Cổng license: chỉ phục vụ camera đã khởi tạo với service_type + license_key
        if not registry.is_allowed(stream_id):
            context.set_code(grpc.StatusCode.PERMISSION_DENIED)
            context.set_details(
                f"stream_id không có license_key/service_type '{settings.SERVICE_TYPE}'")
            return detector_pb2.DetectResponse()

        buf = np.frombuffer(request.image, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Cannot decode JPEG")
            return detector_pb2.DetectResponse()

        if img.shape[0] != request.resized_height or img.shape[1] != request.resized_width:
            img = cv2.resize(img, (request.resized_width, request.resized_height),
                             interpolation=cv2.INTER_AREA)

        item = {
            "img": img,
            "orig_w": request.orig_width,
            "orig_h": request.orig_height,
            "allowed_classes": list(request.allowed_classes) or None,
        }
        event = threading.Event()
        req_tuple = {"item": item, "event": event, "result": []}

        self.batch_queue.put(req_tuple)
        if not event.wait(timeout=2.0):
            context.set_code(grpc.StatusCode.DEADLINE_EXCEEDED)
            context.set_details("Batch processing timeout")
            return detector_pb2.DetectResponse()

        all_detections = req_tuple["result"]

        # Đánh giá mask overlap → chỉ trả về vũ khí xác nhận có người cầm
        frame_w = request.orig_width or request.resized_width
        frame_h = request.orig_height or request.resized_height
        confirmed_weapons, should_alarm = self.overlap_detector.evaluate(
            all_detections, stream_id)

        resp = detector_pb2.DetectResponse()
        for w in confirmed_weapons:
            wb = resp.weapons.add()
            wb.x1, wb.y1, wb.x2, wb.y2 = w["x1"], w["y1"], w["x2"], w["y2"]
            wb.confidence = w["confidence"]
            wb.class_name = w["class_name"]
            wb.held_by_person = w.get("held_by_person", True)

        if should_alarm and confirmed_weapons:
            resp.alarm = True
            cam = registry.get(stream_id) or {}
            weapons_info = [{
                "class_name": w["class_name"],
                "confidence": round(float(w["confidence"]), 4),
                "bbox": [w["x1"], w["y1"], w["x2"], w["y2"]],
                "overlap_pixels": w.get("overlap_pixels", 0),
                "overlap_ratio": w.get("overlap_ratio", 0.0),
            } for w in confirmed_weapons]
            # Gửi frame gốc sạch (không vẽ) về CMS — UI sẽ tự vẽ BBox/Mask từ weapons_info
            self.alarm_pool.submit(self._post_alarm, stream_id, weapons_info, img, cam)

        return resp

    def _post_alarm(self, stream_id, weapons_info, frame, cam):
        if not weapons_info:
            return
        h, w = (frame.shape[:2]) if frame is not None else (720, 1280)
        formatted_weapons = []
        boxes_norm = []
        for item in weapons_info:
            b = item.get("bbox", [0, 0, 0, 0])
            norm = [
                round(b[0] / w, 4),
                round(b[1] / h, 4),
                round(b[2] / w, 4),
                round(b[3] / h, 4),
            ]
            w_dict = {
                "class_name": item.get("class_name"),
                "confidence": item.get("confidence"),
                "bbox": b,
                "norm_bbox": norm,
                "overlap_pixels": item.get("overlap_pixels", 0),
                "overlap_ratio": item.get("overlap_ratio", 0.0),
            }
            formatted_weapons.append(w_dict)
            boxes_norm.append(norm)

        first_pixel = formatted_weapons[0]["bbox"] if formatted_weapons else []
        first_norm = formatted_weapons[0]["norm_bbox"] if formatted_weapons else []

        try:
            create_alarm(
                stream_id=stream_id,
                attributes={
                    "camera_name": cam.get("name"),
                    "image_width": w,
                    "image_height": h,
                    "weapon_count": len(formatted_weapons),
                    "object_count": len(formatted_weapons),
                    "weapons": formatted_weapons,
                    "objects": formatted_weapons,
                    "pixel_bbox": first_pixel,
                    "bbox": first_norm,
                    "coordinates": {
                        "bbox": first_norm,
                        "boxes": boxes_norm,
                    },
                },
                frame=frame,
            )
        except Exception as e:
            logger.error("POST /api/alarms lỗi (stream=%s): %s", stream_id, e)

    def stop(self):
        self.shutdown_flag.set()
        self.alarm_pool.shutdown(wait=False)


def serve():
    if registry.refresh() <= 0:
        logger.warning("Chưa có camera nào mang service_type '%s' kèm license_key. "
                       "Service vẫn chạy và poll lại mỗi %ds, nhưng DetectFrame sẽ "
                       "bị từ chối tới khi camera được khởi tạo.",
                       settings.SERVICE_TYPE, settings.POLL_INTERVAL)
    registry.start()

    servicer = YoloWeaponServiceServicer()

    from services.stream_runner import StreamRunnerManager
    runner_mgr = StreamRunnerManager(servicer)
    runner_mgr.start()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
    detector_pb2_grpc.add_YoloWeaponServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{settings.GRPC_PORT}")
    server.start()
    logger.info("gRPC on port %d | service_type=%s | cooldown=%.0fs | %d camera",
                settings.GRPC_PORT, settings.SERVICE_TYPE,
                settings.ALARM_COOLDOWN, registry.count())
    try:
        server.wait_for_termination()
    finally:
        runner_mgr.stop()
        servicer.stop()
        registry.stop()


if __name__ == "__main__":
    serve()
