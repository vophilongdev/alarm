import os


def _parse_class_names(raw: str) -> dict:
    """CLASS_NAMES="0:person,1:dao,2:sung_ngan" → {0: "person", 1: "dao", 2: "sung_ngan"}."""
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


def _parse_int_list(raw: str) -> list:
    """PERSON_CLASS_IDS="0" hoặc "0,1" → [0] hoặc [0, 1]."""
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            continue
    return out


class Settings:
    """GRPC_DET_WEAPON — YOLO-seg phát hiện vũ khí (port 50061)."""
    # gRPC
    GRPC_PORT = int(os.getenv("GRPC_PORT", 50061))
    LOG_TAG = os.getenv("LOG_TAG", "DET-WEAPON")

    # Paths
    ROOT_PATH = os.getenv("ROOT_PATH", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    MODEL_PT = os.getenv("MODEL_PT", os.path.join(ROOT_PATH, "models", "yolo", "pytorch", "best.pt"))
    MODEL_ENGINE = os.getenv("MODEL_ENGINE", os.path.join(ROOT_PATH, "models", "yolo", "trt", "weapon.engine"))

    # Inference
    USE_TENSORRT = os.getenv("USE_TENSORRT", "true").lower() in ("true", "1", "yes")
    CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", 0.3))
    EXPORT_DYNAMIC_BATCH = os.getenv("EXPORT_DYNAMIC_BATCH", "true").lower() in ("true", "1", "yes")

    # Classes — best.pt: yolov8s-seg, 7 classes (0..6), tên placeholder '0'..'6'.
    # Đặt CLASS_NAMES trong env khi biết nhãn thật.
    CLASS_NAMES = _parse_class_names(os.getenv("CLASS_NAMES", "")) or {
        i: f"obj_{i}" for i in range(7)
    }

    # Mask overlap — class phân loại nội bộ
    # Person: chỉ dùng nội bộ để kiểm tra giao thoa, KHÔNG hiển thị lên UI
    PERSON_CLASS_IDS = _parse_int_list(os.getenv("PERSON_CLASS_IDS", "0"))
    # Weapon: các class vũ khí — duy nhất được đưa lên UI/CMS
    WEAPON_CLASS_IDS = _parse_int_list(os.getenv("WEAPON_CLASS_IDS", "1,2,3,4,5,6"))

    # Dilation & Overlap thresholds
    DILATION_KERNEL_SIZE = int(os.getenv("DILATION_KERNEL_SIZE", 5))
    MIN_OVERLAP_PIXELS = int(os.getenv("MIN_OVERLAP_PIXELS", 15))
    MIN_OVERLAP_RATIO = float(os.getenv("MIN_OVERLAP_RATIO", 0.05))

    # CMS
    BASE_URL = os.getenv("BASE_URL_API", "")
    SESSION_KEY = os.getenv("SESSION_KEY", "")

    # Camera được khởi tạo theo service_type của license_key.
    SERVICE_TYPE = os.getenv("SERVICE_TYPE", "weapon,weapon_detection")
    POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", 20))

    # Alarm
    ALARM_TYPE = os.getenv("ALARM_TYPE", "weapon")
    ALARM_SOURCE = os.getenv("ALARM_SOURCE", "grpc_det_weapon")
    JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", 60))

    # Cooldown — chống spam alarm cho cùng 1 camera
    ALARM_COOLDOWN = float(os.getenv("ALARM_COOLDOWN", 10.0))

    # Batch
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 8))
    BATCH_TIMEOUT_MS = float(os.getenv("BATCH_TIMEOUT_MS", 10))


settings = Settings()
