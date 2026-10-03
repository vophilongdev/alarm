"""
alarm_client.py — client CMS cho service lấn chiếm.

Hai việc:
  1) GET  /api/cameras  → lọc camera theo service_type lấy từ license_key
  2) POST /api/alarms   → bắn alarm lấn chiếm

Body POST /api/alarms theo đúng schema AlarmCreate:
    stream_id, type, source, attributes, time, image_base64
Các field id / created_at / from_device_* / rule do backend tự sinh.
"""
import base64
import logging
import threading
from datetime import datetime, timedelta, timezone

import cv2
import requests
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from core.settings import settings

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logger = logging.getLogger(__name__)

_TZ_VN = timezone(timedelta(hours=7))
_LOCK = threading.Lock()

_session = requests.Session()
_session.mount("https://", HTTPAdapter(
    max_retries=Retry(total=2, backoff_factor=0.3,
                      status_forcelist=(500, 502, 503, 504),
                      allowed_methods=("GET", "POST")),
    pool_connections=8, pool_maxsize=16,
))


def _headers() -> dict:
    return {"session": settings.SESSION_KEY, "Content-Type": "application/json"}


def _svc_types(cam: dict) -> list:
    """service_type của camera; nhận cả list và chuỗi phân tách bằng dấu phẩy."""
    raw = cam.get("service_type") or []
    if isinstance(raw, str):
        raw = [t.strip() for t in raw.split(",") if t.strip()]
    return raw if isinstance(raw, list) else []


def _fetch_rules_map() -> dict:
    """Lấy danh sách /api/rules để tự động ghép parameter.zone nếu cam.parameter là null."""
    try:
        url = f"{settings.BASE_URL.rstrip('/')}/api/rules"
        resp = _session.get(url, headers=_headers(), verify=False, timeout=(5.0, 10.0))
        resp.raise_for_status()
        rules_map = {}
        for rule in resp.json().get("data", []):
            rule_type = rule.get("type") or "encroach"
            for flt in rule.get("filters", []):
                sid = flt.get("stream_id")
                cond = flt.get("condition")
                status = flt.get("status", 1)
                if sid and cond:
                    if sid not in rules_map:
                        rules_map[sid] = {}
                    rules_map[sid][rule_type] = {"status": status, "zone": cond}
                    if rule_type == "encroach":
                        rules_map[sid]["sidewalk_occupation"] = {"status": status, "zone": cond}
        return rules_map
    except Exception as e:
        logger.warning("Không thể lấy /api/rules: %s", e)
        return {}


def get_cameras() -> list:
    """
    Lấy camera của service lấn chiếm.
    Camera được khởi tạo theo service_type của license_key: chỉ giữ camera
    có license_key và service_type chứa settings.SERVICE_TYPE.
    """
    url = f"{settings.BASE_URL.rstrip('/')}/api/cameras"
    resp = _session.get(url, headers=_headers(), verify=False, timeout=(10.0, 20.0))
    resp.raise_for_status()

    rules_map = _fetch_rules_map()

    out = []
    for cam in resp.json().get("data", []):
        license_key = cam.get("license_key") or []
        if isinstance(license_key, str):
            license_key = [license_key]
        if not license_key:
            continue
        target_types = set(t.strip() for t in settings.SERVICE_TYPE.split(",") if t.strip())
        cam_types = set(_svc_types(cam))
        if not target_types.intersection(cam_types):
            continue

        cid = str(cam.get("id"))
        param = cam.get("parameter") or {}
        rules_param = rules_map.get(cid) or {}
        final_param = {**rules_param, **param} if param else rules_param

        out.append({
            "id": cam.get("id"),
            "name": cam.get("name"),
            "status": cam.get("status"),
            "url": cam.get("url"),
            "service_type": _svc_types(cam),
            "license_key": license_key,
            "parameter": final_param,
            "storage_url": cam.get("storage_url"),
            "storage_port": cam.get("storage_port"),
            "storage_username": cam.get("storage_username"),
            "storage_password": cam.get("storage_password"),
            "storage_channel": cam.get("storage_channel"),
            "use_sdk": cam.get("use_sdk", 0),
            "manufacturer": cam.get("manufacturer") or "dahua",
        })

    logger.info("Tìm thấy %d camera cho service_type=%s", len(out), settings.SERVICE_TYPE)
    return out


def create_alarm(stream_id: str, attributes: dict = None, frame=None) -> dict:
    """
    POST /api/alarms — báo lấn chiếm.
    frame: ảnh BGR làm bằng chứng, encode JPEG → image_base64.
    """
    import uuid
    attrs = attributes or {}
    attrs.setdefault("frame_id", str(uuid.uuid4()))

    payload = {
        "stream_id": stream_id,
        "type": settings.ALARM_TYPE,
        "source": stream_id,
        "attributes": attrs,
        "time": datetime.now(_TZ_VN).timestamp(),
    }

    if frame is not None:
        h, w = frame.shape[:2]
        if w > 1280 or h > 720:
            scale = min(1280 / w, 720 / h)
            nw, nh = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)

        ok, buf = cv2.imencode(".jpg", frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), settings.JPEG_QUALITY])
        if ok:
            payload["image_base64"] = base64.b64encode(buf.tobytes()).decode("ascii")

    url = f"{settings.BASE_URL.rstrip('/')}/api/alarms"
    with _LOCK:
        resp = _session.post(url, json=payload, headers=_headers(),
                             verify=False, timeout=(10.0, 20.0))
    logger.info("STATUS-CREATE-ALARM: %s | stream=%s", resp.status_code, stream_id)
    resp.raise_for_status()
    return resp.json()


def patch_camera(stream_id: str, online: int) -> bool:
    """
    Cập nhật trạng thái camera trên CMS (online = 1 hoặc 0).
    Gọi PATCH /api/cameras/{stream_id} với {"online": online}
    """
    url = f"{settings.BASE_URL.rstrip('/')}/api/cameras/{stream_id}"
    try:
        with _LOCK:
            resp = _session.patch(
                url,
                json={"online": online},
                headers=_headers(),
                verify=False,
                timeout=(3.0, 5.0),
            )
        logger.info("STATUS-PATCH-CAMERA: %s | stream=%s | online=%d",
                    resp.status_code, stream_id, online)
        return resp.status_code == 200
    except Exception as e:
        logger.warning("Lỗi patch_camera (stream=%s, online=%d): %s",
                       stream_id, online, e)
        return False

