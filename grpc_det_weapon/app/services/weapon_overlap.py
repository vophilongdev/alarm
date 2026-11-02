"""
weapon_overlap.py — Thuật toán giao thoa mặt nạ Người–Vũ Khí.

Quy trình 4 bước:
  1) Tách danh sách detection thành tập Person (P) và Weapon (W).
  2) Lọc nhanh: BBox(P_i) có giao BBox(W_j) không? (tránh tính mask nặng).
  3) Dilate nhẹ mask người, Bitwise AND với mask vũ khí → đếm pixel giao.
  4) Nếu overlap đạt ngưỡng → xác nhận vũ khí đang được cầm → kích hoạt alarm.

Đầu ra: CHỈ trả về danh sách vũ khí (bỏ người), đảm bảo UI chỉ nhận được
đối tượng vũ khí cần cảnh báo.
"""
import logging
import threading
import time

import cv2
import numpy as np

from core.settings import settings

logger = logging.getLogger(__name__)


def _bbox_intersects(a, b) -> bool:
    """Kiểm tra nhanh 2 bounding box có giao nhau không (toạ độ x1,y1,x2,y2)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    if ax1 >= bx2 or bx1 >= ax2:
        return False
    if ay1 >= by2 or by1 >= ay2:
        return False
    return True


def _compute_mask_overlap(person_mask, weapon_mask, kernel_size: int) -> tuple:
    """
    Tính diện tích giao thoa giữa mask người (đã dilate) và mask vũ khí.

    Returns: (overlap_pixels, overlap_ratio)
      overlap_pixels: số pixel giao nhau
      overlap_ratio:  tỷ lệ so với diện tích mask vũ khí
    """
    if person_mask is None or weapon_mask is None:
        return 0, 0.0

    # Đảm bảo cùng kích thước
    if person_mask.shape != weapon_mask.shape:
        weapon_mask = cv2.resize(weapon_mask, (person_mask.shape[1], person_mask.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)

    # Dilate mask người để mở rộng phạm vi tiếp xúc tay/thân
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    person_dilated = cv2.dilate(person_mask, kernel, iterations=1)

    # Bitwise AND → vùng giao nhau
    overlap = np.logical_and(person_dilated > 0, weapon_mask > 0)
    overlap_pixels = int(np.sum(overlap))

    weapon_area = int(np.sum(weapon_mask > 0))
    overlap_ratio = overlap_pixels / weapon_area if weapon_area > 0 else 0.0

    return overlap_pixels, overlap_ratio


class WeaponOverlapDetector:
    """
    Phát hiện vũ khí đang được người cầm dựa trên giao thoa segmentation mask.

    Đầu ra chỉ chứa danh sách VŨ KHÍ — class Người chỉ dùng nội bộ.
    """

    def __init__(self):
        self._cooldowns = {}          # stream_id → last_alarm_monotonic
        self._cooldown_lock = threading.Lock()

    def evaluate(self, detections: list, stream_id: str) -> tuple:
        """
        Đánh giá danh sách detections (bbox + mask) của một frame.

        Args:
            detections: list of dict {x1,y1,x2,y2, class_id, class_name,
                                      confidence, mask (np.ndarray|None)}
            stream_id:  uuid camera

        Returns:
            (confirmed_weapons, should_alarm)
            confirmed_weapons: list[dict] — CHỈ các vũ khí đã xác nhận có người cầm
            should_alarm: bool — True nếu nên bắn alarm (đã kiểm tra cooldown)
        """
        person_ids = set(settings.PERSON_CLASS_IDS)
        weapon_ids = set(settings.WEAPON_CLASS_IDS)

        persons = [d for d in detections if d["class_id"] in person_ids]
        weapons = [d for d in detections if d["class_id"] in weapon_ids]

        if not persons or not weapons:
            return [], False

        kernel_size = settings.DILATION_KERNEL_SIZE
        min_pixels = settings.MIN_OVERLAP_PIXELS
        min_ratio = settings.MIN_OVERLAP_RATIO

        confirmed = []
        for w in weapons:
            w_bbox = (w["x1"], w["y1"], w["x2"], w["y2"])
            for p in persons:
                p_bbox = (p["x1"], p["y1"], p["x2"], p["y2"])

                # Bước 2: Lọc nhanh BBox
                if not _bbox_intersects(p_bbox, w_bbox):
                    continue

                # Bước 3: Mask overlap chính xác (pixel-level)
                overlap_px, overlap_ratio = _compute_mask_overlap(
                    p["mask"], w["mask"], kernel_size)

                if overlap_px >= min_pixels and overlap_ratio >= min_ratio:
                    confirmed.append({
                        **w,
                        "held_by_person": True,
                        "overlap_pixels": overlap_px,
                        "overlap_ratio": round(overlap_ratio, 4),
                    })
                    logger.info(
                        "[%s] 🔫 Vũ khí xác nhận: class=%s conf=%.2f "
                        "overlap=%dpx (%.1f%%)",
                        stream_id, w["class_name"], w["confidence"],
                        overlap_px, overlap_ratio * 100)
                    break  # 1 vũ khí chỉ cần match 1 người

        if not confirmed:
            return [], False

        # Bước 4: Kiểm tra cooldown
        should_alarm = self._check_cooldown(stream_id)
        return confirmed, should_alarm

    def _check_cooldown(self, stream_id: str) -> bool:
        now = time.monotonic()
        with self._cooldown_lock:
            last = self._cooldowns.get(stream_id, 0.0)
            if now - last > settings.ALARM_COOLDOWN:
                self._cooldowns[stream_id] = now
                return True
        return False
