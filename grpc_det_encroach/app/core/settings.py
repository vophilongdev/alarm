import os


def _parse_class_names(raw: str) -> dict:
    """CLASS_NAMES="0:nguoi,1:xe_may" → {0: "nguoi", 1: "xe_may"}."""
    mapping = {}
    for part in raw.split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        idx, name = part.split(":", 1)
        try:
            mapping[int(idx.strip())] = name.strip()
        except ValueError:
            continue
    return mapping


class Settings:
    """GRPC_DET_ENCROACH — YOLO detector lấn chiếm lòng lề đường (port 50060)."""
    # gRPC
    GRPC_PORT = int(os.getenv("GRPC_PORT", 50060))
    LOG_TAG = os.getenv("LOG_TAG", "DET-ENCROACH")

    # Paths
    ROOT_PATH = os.getenv("ROOT_PATH", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    MODEL_PT = os.getenv("MODEL_PT", os.path.join(ROOT_PATH, "models", "yolo", "pytorch", "best.pt"))
    MODEL_ENGINE = os.getenv("MODEL_ENGINE", os.path.join(ROOT_PATH, "models", "yolo", "trt", "encroach.engine"))

    # Inference
    USE_TENSORRT = os.getenv("USE_TENSORRT", "true").lower() in ("true", "1", "yes")
    CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", 0.2))
    EXPORT_DYNAMIC_BATCH = os.getenv("EXPORT_DYNAMIC_BATCH", "true").lower() in ("true", "1", "yes")

    # Classes — best.pt xuất từ dataset Lan_chiem3 (roboflow) với tên class placeholder
    # '0'..'9'; đặt CLASS_NAMES trong env khi biết nhãn thật.
    CLASS_NAMES = _parse_class_names(os.getenv("CLASS_NAMES", "")) or {
        i: f"obj_{i}" for i in range(10)
    }

    # CMS
    BASE_URL = os.getenv("BASE_URL_API", "")
    SESSION_KEY = os.getenv("SESSION_KEY", "")

    # Camera được khởi tạo theo service_type của license_key. Service này chỉ
    # mở khi có camera mang service_type dưới đây kèm license_key hợp lệ.
    SERVICE_TYPE = os.getenv("SERVICE_TYPE", "encroach")
    POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", 20))

    # Alarm
    ALARM_TYPE = os.getenv("ALARM_TYPE", "encroach")
    ALARM_SOURCE = os.getenv("ALARM_SOURCE", "grpc_det_encroach")
    JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", 60))

    # Lấn chiếm: quá DWELL_SECONDS giây liên tục trong ROI thì báo
    DWELL_SECONDS = float(os.getenv("DWELL_SECONDS", 10.0))
    ALARM_COOLDOWN = float(os.getenv("ALARM_COOLDOWN", 60.0))

    # Tracker
    MATCH_IOU_THRESHOLD = float(os.getenv("MATCH_IOU_THRESHOLD", 0.25))
    MAX_LOST_ROUNDS = int(os.getenv("MAX_LOST_ROUNDS", 60))
    TRACK_TTL = float(os.getenv("TRACK_TTL", 60.0))


settings = Settings()
