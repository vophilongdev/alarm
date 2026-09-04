"""
server.py — gRPC service lấn chiếm vỉa hè (GRPC_DET_ENCROACH).

Luồng một request:
    DetectFrame(image, roi, stream_id)
      → kiểm tra stream_id có license_key + service_type hợp lệ
      → YOLO detect (dynamic batching)
      → tracker: bbox nào nằm trong ROI, nằm liên tục bao lâu
      → dwell > DWELL_SECONDS  →  POST /api/alarms

ROI do UI vẽ và gửi kèm request; service không tự sinh ROI. Service chỉ mở khi
CMS có camera mang service_type của license_key tương ứng.
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
from services.encroach_tracker import get_tracker, parse_roi
from services.yolo_model import init_yolo, detect_batch

BATCH_SIZE = int(os.getenv("BATCH_SIZE", 8))
BATCH_TIMEOUT_SEC = float(os.getenv("BATCH_TIMEOUT_MS", 10)) / 1000.0

logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s [{settings.LOG_TAG}] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


class YoloServiceServicer(detector_pb2_grpc.YoloServiceServicer):
    def __init__(self):
        self.model, self.device, self.model_is_pt = init_yolo()
        self.batch_queue = queue.Queue()
        self.shutdown_flag = threading.Event()
        self.alarm_pool = futures.ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="alarm")

        self.worker_thread = threading.Thread(target=self._batch_worker_loop, daemon=True)
        self.worker_thread.start()
        logger.info("Dynamic batching bật (BATCH_SIZE=%d, TIMEOUT=%.0fms). Model sẵn sàng.",
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
                batch_results = detect_batch(
                    model=self.model,
                    device=self.device,
                    model_is_pt=self.model_is_pt,
                    batch_items=[r["item"] for r in batch_requests],
                )
            except Exception as e:
                logger.error("Batch detect lỗi: %s", e)
                if self.device != "cpu":
                    torch.cuda.empty_cache()
                batch_results = [[] for _ in batch_requests]

            t_gpu_dur_ms = (time.perf_counter() - t_gpu_start) * 1000.0

            total_detected = 0
            class_counts = Counter()
            for idx, req_tuple in enumerate(batch_requests):
                bboxes = batch_results[idx]
                req_tuple["result"] = bboxes
                for box in bboxes:
                    total_detected += 1
                    class_counts[box["class_name"]] += 1
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

        detections = [{
            "bbox": (b["x1"], b["y1"], b["x2"], b["y2"]),
            "class_name": b["class_name"],
            "confidence": b["confidence"],
        } for b in req_tuple["result"]]

        # ROI từ UI, quy về pixel của khung gốc
        frame_w = request.orig_width or request.resized_width
        frame_h = request.orig_height or request.resized_height
        roi_contour = parse_roi([(p.x, p.y) for p in request.roi], frame_w, frame_h)

        tracker = get_tracker(stream_id)
        results, alarm_ids = tracker.update(detections, roi_contour)

        resp = detector_pb2.DetectResponse()
        for r in results:
            x1, y1, x2, y2 = r["bbox"]
            bb = resp.bboxes.add()
            bb.x1, bb.y1, bb.x2, bb.y2 = x1, y1, x2, y2
            bb.confidence = r["confidence"]
            bb.class_name = r["class_name"]
            bb.track_id = r["track_id"]
            bb.in_roi = r["in_roi"]
            bb.dwell_seconds = r["dwell_seconds"]

        if alarm_ids:
            resp.alarm = True
            resp.alarm_track_ids.extend(alarm_ids)
            # Một frame nhiều vật thể vi phạm vẫn chỉ là một alarm: sự kiện lấn
            # chiếm là cả khung, không phải từng vật thể. Tránh spam CMS.
            alarm_set = set(alarm_ids)
            tracks = [r for r in results if r["track_id"] in alarm_set]
            evidence = self._draw_evidence(img, results, roi_contour,
                                           frame_w, frame_h, alarm_ids)
            self.alarm_pool.submit(self._post_alarm, stream_id, tracks, evidence)

        return resp

    @staticmethod
    def _draw_evidence(img, results, roi_contour, frame_w, frame_h, alarm_ids):
        """Vẽ ROI + bbox vi phạm lên ảnh bằng chứng, ở kích thước khung gốc."""
        try:
            frame = cv2.resize(img, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR) \
                if (img.shape[1], img.shape[0]) != (frame_w, frame_h) else img.copy()
            if roi_contour is not None:
                cv2.polylines(frame, [roi_contour], True, (0, 255, 255), 2)
            alarm_set = set(alarm_ids)
            for r in results:
                if r["track_id"] not in alarm_set:
                    continue
                x1, y1, x2, y2 = r["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{r['class_name']} {r['dwell_seconds']:.0f}s",
                            (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)
            return frame
        except Exception as e:
            logger.warning("Vẽ ảnh bằng chứng lỗi: %s", e)
            return img

    def _post_alarm(self, stream_id, tracks, frame):
        if not tracks:
            return
        cam = registry.get(stream_id) or {}
        objects = [{
            "track_id": t["track_id"],
            "class_name": t["class_name"],
            "confidence": round(float(t["confidence"]), 4),
            "dwell_seconds": round(float(t["dwell_seconds"]), 2),
            "bbox": list(t["bbox"]),
        } for t in tracks]
        try:
            create_alarm(
                stream_id=stream_id,
                attributes={
                    "camera_name": cam.get("name"),
                    "threshold_seconds": settings.DWELL_SECONDS,
                    "object_count": len(objects),
                    "max_dwell_seconds": max(o["dwell_seconds"] for o in objects),
                    "objects": objects,
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

    servicer = YoloServiceServicer()

    from services.stream_runner import StreamRunnerManager
    runner_mgr = StreamRunnerManager(servicer)
    runner_mgr.start()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
    detector_pb2_grpc.add_YoloServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{settings.GRPC_PORT}")
    server.start()
    logger.info("gRPC on port %d | service_type=%s | dwell>%.0fs | %d camera",
                settings.GRPC_PORT, settings.SERVICE_TYPE,
                settings.DWELL_SECONDS, registry.count())
    try:
        server.wait_for_termination()
    finally:
        runner_mgr.stop()
        servicer.stop()
        registry.stop()


if __name__ == "__main__":
    serve()
