"""
encroach_tracker.py — logic lấn chiếm.

ROI do UI vẽ và gửi kèm request; service chỉ kiểm tra vật thể có nằm trong ROI
và nằm liên tục bao lâu. Quá DWELL_SECONDS thì báo alarm.

Thời gian đếm bằng wall-clock (time.monotonic), không đếm theo số frame, để
ngưỡng 10 giây không bị lệch khi FPS thay đổi.
"""
import logging
import threading
import time

import cv2
import numpy as np

from core.settings import settings

logger = logging.getLogger(__name__)


def parse_roi(points, frame_w: int, frame_h: int):
    """
    points: list (x, y) từ UI. Toạ độ chuẩn hoá (<=1.0) hoặc pixel.
    Trả contour int32 cho pointPolygonTest, hoặc None nếu ROI không hợp lệ.
    """
    if not points or len(points) < 3:
        return None

    vals = [c for p in points for c in (p[0], p[1])]
    normalized = all(0.0 <= v <= 1.0 for v in vals)

    pts = []
    for x, y in points:
        if normalized:
            pts.append([int(round(x * frame_w)), int(round(y * frame_h))])
        else:
            pts.append([int(round(x)), int(round(y))])
    return np.array(pts, dtype=np.int32)


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class _Track:
    __slots__ = ("track_id", "bbox", "class_name", "confidence",
                 "roi_since", "last_seen", "alarmed_at", "lost")

    def __init__(self, track_id: int, bbox, class_name: str, confidence: float, now: float):
        self.track_id = track_id
        self.bbox = bbox
        self.class_name = class_name
        self.confidence = confidence
        self.roi_since = None      # mốc bắt đầu vào ROI, None = đang ngoài ROI
        self.last_seen = now
        self.alarmed_at = None     # lần bắn alarm gần nhất, None = chưa từng báo
        self.lost = 0

    def dwell(self, now: float) -> float:
        return 0.0 if self.roi_since is None else now - self.roi_since

    def can_alarm(self, now: float) -> bool:
        """Lần đầu vượt ngưỡng thì báo ngay; sau đó phải hết cooldown."""
        return self.alarmed_at is None or now - self.alarmed_at > settings.ALARM_COOLDOWN


class EncroachTracker:
    """Một instance cho mỗi stream_id. Không chia sẻ state giữa các camera."""

    def __init__(self, stream_id: str):
        self.stream_id = stream_id
        self._tracks = {}
        self._next_id = 1
        self._lock = threading.Lock()

    def _match(self, bbox, used: set):
        """Ghép detection với track cũ bằng IoU lớn nhất."""
        best_id, best_iou = None, 0.0
        for tid, tr in self._tracks.items():
            if tid in used:
                continue
            v = _iou(bbox, tr.bbox)
            if v > best_iou:
                best_id, best_iou = tid, v
        return best_id if best_iou >= settings.MATCH_IOU_THRESHOLD else None

    def update(self, detections, roi_contour, now: float = None):
        """
        detections: list dict {bbox:(x1,y1,x2,y2), class_name, confidence}
        roi_contour: contour từ parse_roi, None nghĩa là camera chưa vẽ ROI.

        Trả (results, alarm_ids):
          results  - mỗi detection kèm track_id / in_roi / dwell_seconds
          alarm_ids - track vừa vượt ngưỡng, cần bắn alarm
        """
        now = now if now is not None else time.monotonic()
        results, alarm_ids, used = [], [], set()

        with self._lock:
            for det in detections:
                bbox = det["bbox"]
                tid = self._match(bbox, used)

                if tid is None:
                    tid = self._next_id
                    self._next_id += 1
                    tr = _Track(tid, bbox, det["class_name"], det["confidence"], now)
                    self._tracks[tid] = tr
                else:
                    tr = self._tracks[tid]
                    tr.bbox = bbox
                    tr.class_name = det["class_name"]
                    tr.confidence = det["confidence"]
                used.add(tid)

                tr.last_seen = now
                tr.lost = 0

                in_roi = self._in_roi(bbox, roi_contour)
                if in_roi:
                    if tr.roi_since is None:
                        tr.roi_since = now
                else:
                    tr.roi_since = None
                    tr.alarmed_at = None  # ra khỏi ROI thì reset, lần sau vào tính lại

                dwell = tr.dwell(now)
                if in_roi and dwell > settings.DWELL_SECONDS and tr.can_alarm(now):
                    tr.alarmed_at = now
                    alarm_ids.append(tid)
                    logger.info("[%s] Lấn chiếm: track=%d class=%s dwell=%.1fs",
                                self.stream_id, tid, tr.class_name, dwell)

                results.append({
                    "bbox": bbox,
                    "class_name": tr.class_name,
                    "confidence": tr.confidence,
                    "track_id": tid,
                    "in_roi": in_roi,
                    "dwell_seconds": dwell,
                })

            self._prune(used, now)

        return results, alarm_ids

    @staticmethod
    def _in_roi(bbox, roi_contour) -> bool:
        """Không có ROI thì không có lấn chiếm — vùng cấm do UI quyết định."""
        if roi_contour is None:
            return False
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        return cv2.pointPolygonTest(roi_contour, (float(cx), float(cy)), False) >= 0

    def _prune(self, used: set, now: float):
        for tid in list(self._tracks):
            if tid in used:
                continue
            tr = self._tracks[tid]
            tr.lost += 1
            if tr.lost > settings.MAX_LOST_ROUNDS or now - tr.last_seen > settings.TRACK_TTL:
                del self._tracks[tid]

    def get(self, track_id: int):
        return self._tracks.get(track_id)


_registry = {}
_registry_lock = threading.Lock()


def get_tracker(stream_id: str) -> EncroachTracker:
    with _registry_lock:
        tr = _registry.get(stream_id)
        if tr is None:
            tr = EncroachTracker(stream_id)
            _registry[stream_id] = tr
        return tr
