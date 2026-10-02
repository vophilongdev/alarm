# Dockerfile — YOLO gRPC detector service (unified).
# Build from the SERVICE directory root:
#   docker build -t harbor.tado.vn/long_dev/grpc-det-encroach:v1.0.0 .
FROM harbor.tado.vn/core-ai-trafsec/base:cu121-trt10
# prod: FROM harbor.tado.vn/core-ai-trafsec/base:cu121-trt10

WORKDIR /app
COPY app/ /app/

# Đóng gói ENV mặc định trực tiếp vào Image
ENV GRPC_PORT=50060 \
    USE_TENSORRT=true \
    CONF_THRESHOLD=0.2 \
    BATCH_SIZE=8 \
    BATCH_TIMEOUT_MS=10 \
    BASE_URL_API=https://api-dev.tado.vn \
    SESSION_KEY=359c8b594fe339f1d7b8337ad22f4ac1 \
    SERVICE_TYPE=sidewalk_occupation \
    POLL_INTERVAL=20 \
    DWELL_SECONDS=60 \
    ALARM_COOLDOWN=60 \
    ALARM_TYPE=encroach \
    ALARM_SOURCE=grpc_det_encroach \
    JPEG_QUALITY=60

ARG PORT=50060
EXPOSE ${PORT}
CMD ["python3", "server.py"]

