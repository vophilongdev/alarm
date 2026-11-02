# Dockerfile — YOLO-seg gRPC weapon detector service.
# Build from the SERVICE directory root:
#   docker build -t harbor.tado.vn/long_dev/grpc-det-weapon:v1.0.0 .
FROM harbor.tado.vn/core-ai-trafsec/base:cu121-trt10

WORKDIR /app
COPY app/ /app/

# Đóng gói ENV mặc định trực tiếp vào Image
ENV GRPC_PORT=50061 \
    USE_TENSORRT=true \
    CONF_THRESHOLD=0.3 \
    BATCH_SIZE=8 \
    BATCH_TIMEOUT_MS=10 \
    BASE_URL_API=https://api-dev.tado.vn \
    SESSION_KEY=359c8b594fe339f1d7b8337ad22f4ac1 \
    SERVICE_TYPE=weapon,weapon_detection \
    POLL_INTERVAL=20 \
    ALARM_COOLDOWN=10 \
    ALARM_TYPE=weapon \
    ALARM_SOURCE=grpc_det_weapon \
    JPEG_QUALITY=60 \
    PERSON_CLASS_IDS=0 \
    WEAPON_CLASS_IDS=1,2,3,4,5,6 \
    DILATION_KERNEL_SIZE=5 \
    MIN_OVERLAP_PIXELS=15 \
    MIN_OVERLAP_RATIO=0.05

ARG PORT=50061
EXPOSE ${PORT}
CMD ["python3", "server.py"]
