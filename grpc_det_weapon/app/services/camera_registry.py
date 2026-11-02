"""
camera_registry.py — cổng kiểm soát service phát hiện vũ khí.

Registry poll GET /api/cameras định kỳ, giữ danh sách stream_id được phép,
và DetectFrame từ chối stream không nằm trong danh sách.
"""
import logging
import threading

from core.settings import settings
from services.alarm_client import get_cameras

logger = logging.getLogger(__name__)

class CameraRegistry:
    def __init__(self):
        self._cameras = {}          # stream_id -> cam dict
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._ready = threading.Event()

    def refresh(self) -> int:
        """Poll CMS. Trả số camera được phép."""
        try:
            cams = get_cameras()
        except Exception as e:
            logger.error("Poll /api/cameras lỗi: %s", e)
            return -1

        allowed = {}
        for cam in cams:
            cam_id = cam.get("id")
            if not cam_id:
                continue
            if cam.get("status") in (0, "0", False, "false"):
                logger.info("Bỏ camera %s: status tắt", cam_id)
                continue
            allowed[str(cam_id)] = cam

        with self._lock:
            added = set(allowed) - set(self._cameras)
            removed = set(self._cameras) - set(allowed)
            self._cameras = allowed

        for cid in added:
            logger.info("Mở camera %s (%s) license_key=%d key(s)",
                        cid, allowed[cid].get("name"), len(allowed[cid]["license_key"]))
        for cid in removed:
            logger.info("Đóng camera %s: hết license_key hoặc service_type", cid)

        self._ready.set()
        return len(allowed)

    def is_allowed(self, stream_id: str) -> bool:
        if not stream_id:
            return False
        with self._lock:
            return str(stream_id) in self._cameras

    def get(self, stream_id: str):
        with self._lock:
            return self._cameras.get(str(stream_id))

    def count(self) -> int:
        with self._lock:
            return len(self._cameras)

    def wait_ready(self, timeout: float = None) -> bool:
        return self._ready.wait(timeout)

    def start(self):
        self.refresh()
        self._thread = threading.Thread(target=self._loop, name="camera-poll", daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.wait(settings.POLL_INTERVAL):
            self.refresh()

    def stop(self):
        self._stop.set()


registry = CameraRegistry()
