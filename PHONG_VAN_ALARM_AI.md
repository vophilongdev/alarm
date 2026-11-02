# 🎯 CẨM NANG PHỎNG VẤN CHUYÊN SÂU: HỆ THỐNG DỊCH VỤ CẢNH BÁO ALARM AI
### (Chuyên biệt: Phát hiện Vũ khí - grpc_det_weapon & Phát hiện Lấn chiếm - grpc_det_encroach)

---

## 📌 PHẦN 1: CÂU HỎI TRỌNG TÂM — "TẠI SAO BẠN BIẾT ĐÂY LÀ KIẾN TRÚC MICROSERVICES?"

Khi Nhà tuyển dụng hỏi: *"Nhìn vào dự án Alarm này, tại sao bạn khẳng định đây là hệ thống Microservices chứ không phải Monolith?"*, hãy trình bày **5 luận điểm kỹ thuật chuẩn Senior** dựa trên kiến trúc thực tế:

```
                  ┌────────────────────────────────────────────────────────┐
                  │                 HỆ THỐNG CMS BACKEND                   │
                  │   REST API (/api/alarms, /api/cameras, /api/rules)     │
                  └───────────▲───────────────────────────────▲────────────┘
                              │                               │
                              │ POST /api/alarms              │ POST /api/alarms
                              │ (Clean Frame + Norm BBox)     │ (Dwell Violation Evidence)
                              │                               │
┌─────────────────────────────┴─────────────────┐ ┌───────────┴───────────────────────────────┐
│ [MICROSERVICE 1]: GRPC_DET_WEAPON             │ │ [MICROSERVICE 2]: GRPC_DET_ENCROACH         │
│ Container: grpc_det_weapon (Port Host: 50062) │ │ Container: grpc_det_encroach (Port: 50063)  │
│                                               │ │                                             │
│ - Ingestion: RTSP (TCP) / Dahua & Hik SDK     │ │ - Ingestion: RTSP (TCP) / Dahua & Hik SDK   │
│ - Dynamic Licensing: CameraRegistry (poll 20s)│ │ - Dynamic Licensing: CameraRegistry         │
│ - AI Model: YOLOv8-seg (Instance Segmentation)│ │ - AI Model: YOLO Detection (Bounding Box)   │
│ - Core Logic: Mask Overlap (Dilation + AND)   │ │ - Core Logic: Polygon ROI + Dwell Time      │
│ - Urgency: Zero Dwell (Báo động tức thì)      │ │ - Urgency: Wall-Clock Dwell Time (>= 10s)   │
│ - Output: CHỈ gửi Vũ khí (ẩn class Người)     │ │ - Output: Vật thể lấn chiếm (xe, bàn ghế)   │
└───────────────────────┬───────────────────────┘ └───────────────────┬─────────────────────────┘
                        │ gRPC / Protobuf (:50061)                    │ gRPC / Protobuf (:50060)
                        │ Image Bytes (detector.proto)                │ Image Bytes + ROI Poly
┌───────────────────────▼───────────────────────┐ ┌───────────────────▼─────────────────────────┐
│ DEDICATED GPU WORKER: WEAPON SEGMENTATION     │ │ DEDICATED GPU WORKER: ENCROACH DETECTION    │
│ - Dynamic Batching (Batch 8, Timeout 10ms)    │ │ - Dynamic Batching (Batch 8, Timeout 10ms)  │
│ - TensorRT 10 FP16 Engine                     │ │ - TensorRT 10 Engine                        │
│ - Dead Request Cancellation                   │ │ - Dead Request Cancellation                 │
└───────────────────────────────────────────────┘ └─────────────────────────────────────────────┘
```

### 5 Bằng chứng kỹ thuật không thể chối cãi:

1. **Phân tách vật lý và độc lập vòng đời triển khai (Independent Containerization):**
   - Mỗi nghiệp vụ an ninh được đóng gói thành một container Docker riêng biệt với image độc lập:
     - Dịch vụ Vũ khí: `harbor.tado.vn/long_dev/grpc-det-weapon:v1.0.0`
     - Dịch vụ Lấn chiếm: `harbor.tado.vn/long_dev/grpc-det-encroach:v1.0.0`
   - Mỗi dịch vụ có file `docker-compose.yml`, file `.env`, port mạng (`50061` vs `50062`), mã nguồn và mô hình AI chuyên biệt (`best.pt` seg vs `encroach_v1.pt` det).
   - Khi cập nhật mô hình vũ khí mới hoặc restart container vũ khí, **dịch vụ lấn chiếm vỉa hè và CMS Backend hoàn toàn không bị ảnh hưởng hay gián đoạn**.

2. **Giao tiếp liên dịch vụ qua RPC nhị phân (gRPC over HTTP/2) thay vì in-process:**
   - Trong Monolith, ứng dụng import trực tiếp model: `result = model.predict(frame)`.
   - Trong hệ thống này, các thành phần gọi nhau qua **gRPC** với hợp đồng dữ liệu nhị phân xác định bởi Protocol Buffers (`services/detector.proto`). Frame ảnh được mã hóa thành JPEG byte stream truyền qua mạng, tách biệt hoàn toàn không gian bộ nhớ giữa Client (tầng đọc stream / gateway) và Server (tầng worker suy luận AI trên GPU).

3. **Tính toán không đồng nhất & Khả năng mở rộng độc lập (Heterogeneous Scalability):**
   - **Tác vụ phát hiện vũ khí** đòi hỏi Instance Segmentation rất nặng về GPU VRAM $ightarrow$ có thể scale trên cụm máy chủ GPU cao cấp (NVIDIA RTX 4090 / A10).
   - **Tác vụ phát hiện lấn chiếm** chủ yếu là Object Detection nhẹ kết hợp Polygon Tracking theo thời gian $ightarrow$ có thể chạy trên GPU tầm trung (T4).
   - Việc tách microservice cho phép cấp phát tài nguyên phần cứng chính xác theo nhu cầu của từng bài toán mà không gây lãng phí.

4. **Tách biệt ranh giới nghiệp vụ (Domain-Driven Design):**
   - `grpc_det_weapon`: Chuyên sâu về an ninh khẩn cấp. Đặc trưng: **Zero Dwell Time** (báo động tức thì), thuật toán **Mask Overlap** (giao thoa giữa người và vũ khí), class Người chỉ là trigger nội bộ và bị ẩn khỏi cảnh báo.
   - `grpc_det_encroach`: Chuyên sâu về trật tự đô thị. Đặc trưng: **Dwell Time** (phải chiếm dụng liên tục $\ge 10	ext{s}$ mới tính vi phạm), thuật toán kiểm tra điểm trong đa giác **Polygon ROI** (`pointPolygonTest`).
   - Hai bài toán có logic xử lý và tính chất cảnh báo trái ngược nhau hoàn toàn, việc tách thành 2 microservices giúp mã nguồn cô lập, dễ bảo trì và mở rộng.

5. **Cơ chế khám phá dịch vụ động & Phân quyền License tập trung (Dynamic Camera Licensing):**
   - Cả 2 service không fix cứng camera cục bộ mà sử dụng module `CameraRegistry` chạy nền poll định kỳ CMS Backend (`GET /api/cameras`).
   - Tự động lọc camera theo đúng `service_type` (ví dụ: `weapon` vs `encroachment`) và `license_key`. Nếu quản trị viên trên CMS cấp phép hoặc thu hồi license của một camera, service tự động khởi tạo hoặc hủy luồng stream camera tương ứng theo thời gian thực mà không cần restart service.

---

## ⚡ PHẦN 2: ELEVATOR PITCH (GIỚI THIỆU DỰ ÁN TRONG 60 GIÂY)

> *"Dự án **Alarm AI Services** của em là một hệ thống phân tích hình ảnh an ninh thời gian thực theo kiến trúc **gRPC Microservices**, tích hợp chặt chẽ với CMS Backend để xử lý 2 bài toán an ninh then chốt:*
>
> 1. ***Dịch vụ Phát hiện Vũ khí (`grpc_det_weapon`)***: *Sử dụng **YOLOv8 Instance Segmentation** trên TensorRT kết hợp thuật toán **Mask Overlap 4 bước (Dilation & Bitwise AND)** để chỉ báo động khi vũ khí thực sự được người cầm trên tay, loại bỏ hoàn toàn báo động giả từ dao thớt trên bàn hay tranh ảnh.*
> 2. ***Dịch vụ Phát hiện Lấn chiếm (`grpc_det_encroach`)***: *Giám sát các vùng đa giác ROI phức tạp từ CMS, ứng dụng cơ chế **Wall-clock Dwell Time** để đo thời gian chiếm dụng thực tế, không bị sai lệch khi camera tụt FPS mạng.*
>
> *Hệ thống sở hữu kiến trúc **Dual-mode Ingestion** (hỗ trợ cả gRPC Server lẫn tự động đọc camera qua RTSP TCP và Dahua/Hikvision C-SDKs), áp dụng cơ chế **Dynamic Batching** và **Clean Evidence** (gửi ảnh gốc sạch kèm metadata chuẩn hóa)."*

---

## 🏆 PHẦN 3: ĐIỂM "ĂN TIỀN" CỐT LÕI CỦA DỊCH VỤ VŨ KHÍ (GRPC_DET_WEAPON)

### 1. Tại sao phải dùng Instance Segmentation (YOLOv8-seg) thay vì Bounding Box?
* **Hạn chế chết người của Bounding Box:** Trong không gian 2D của camera, Bounding Box của người và vũ khí rất dễ giao nhau do góc nhìn phối cảnh (perspective). Một người đi ngang qua con dao đặt trên bàn thu ngân hoặc đứng trước bức tranh vẽ khẩu súng thì 2 bounding box vẫn đè lên nhau $ightarrow$ **Gây báo động giả liên tục (False Alarm Disaster)**.
* **Sức mạnh của Segmentation:** Phân đoạn mặt nạ chính xác tới từng pixel giúp hệ thống biết chính xác biên dạng thực tế của đối tượng.

### 2. Thuật toán Mask Overlap 4 bước độc quyền (`WeaponOverlapDetector`)
*Mã nguồn:* [`weapon_overlap.py`](file:///home/tado/long_dev/alarm/grpc_det_weapon/app/services/weapon_overlap.py)
* **Bước 1 — Tách tập thực thể:** Phân tách detections thành tập `Person` (`class_id = 0`) và tập `Weapon` (`class_id in [1..6]`: dao, súng ngắn, súng trường, kéo, gậy, bom).
* **Bước 2 — Fast BBox Rejection (Lọc nhanh $O(1)$):**
  * Kiểm tra giao nhau sơ bộ bằng Bounding Box (`_bbox_intersects`).
  * Nếu 2 box không chạm nhau $ightarrow$ bỏ qua ngay lập tức. **95% frame bình thường được lọc bỏ tại đây, tiết kiệm tối đa CPU/GPU**.
* **Bước 3 — Dilation & Bitwise AND (Pixel-level):**
  * *Vấn đề:* Bàn tay, ngón tay của người cầm vũ khí rất mảnh nên mô hình segmentation thường bị đứt đoạn hoặc nhận diện thiếu phần tiếp xúc.
  * *Giải pháp:* Làm nở nhẹ mặt nạ người bằng phép biến đổi hình thái học `cv2.dilate` với `kernel_size = 5` hoặc `15`:
    $$	ext{Person}_{	ext{dilated}} = 	ext{cv2.dilate}(	ext{Person}_{	ext{mask}}, 	ext{kernel})$$
  * Thực hiện phép giao logic Bitwise AND giữa mặt nạ người đã nở và mặt nạ vũ khí:
    $$	ext{Overlap} = 	ext{Person}_{	ext{dilated}} \land 	ext{Weapon}_{	ext{mask}}$$
* **Bước 4 — Ngưỡng diện tích kép (Dual-threshold):**
  * Chỉ xác nhận vũ khí khi thỏa mãn đồng thời:
    - $	ext{Số pixel giao nhau} \ge 	ext{MIN\_OVERLAP\_PIXELS}$ (mặc định $15 - 100	ext{px}$).
    - $	ext{Tỷ lệ diện tích giao} \ge 	ext{MIN\_OVERLAP\_RATIO}$ (mặc định $5\%$ diện tích vũ khí).

### 3. Bộ lọc cảnh báo thông minh: Class người là Internal Trigger
* Trong response trả về và cảnh báo gửi lên CMS, **hệ thống loại bỏ hoàn toàn thông tin người**, chỉ trả về thông tin vũ khí (`WeaponBox`).
* Class người chỉ đóng vai trò điều kiện kích hoạt nội bộ. Điều này giúp Web UI của CMS không bị rối mắt bởi hàng loạt bounding box người, tập trung 100% vào mối đe dọa vũ khí.

### 4. Zero Dwell Time & Cooldown thông minh
* Khác với các bài toán đô thị cần quan sát thời gian đỗ, vũ khí là tình huống khẩn cấp đe dọa tính mạng $ightarrow$ **Bắn cảnh báo tức thì ngay ở frame đầu tiên phát hiện (Zero Dwell Time)**.
* Sử dụng `ALARM_COOLDOWN = 10s` (hoặc `60s`) trên từng camera để tránh tình trạng flood hàng trăm cảnh báo trong cùng một sự việc.

### 5. Kiến trúc Clean Evidence (Bằng chứng số nguyên vẹn)
* Hệ thống **tuyệt đối không vẽ đè bounding box hay mask lên ảnh** trước khi gửi về CMS qua `POST /api/alarms`.
* Gửi **frame gốc nguyên bản** kèm metadata JSON tọa độ chuẩn hóa (`norm_bbox`, `confidence`, `overlap_ratio`, `overlap_pixels`).
* *Ý nghĩa thực tế:* Giữ nguyên vẹn tính pháp lý của bằng chứng hình ảnh (không bị nét vẽ che khuất mặt thủ phạm hay chi tiết vũ khí), đồng thời cho phép Web UI CMS bật/tắt hiển thị linh hoạt.

---

## 🏆 PHẦN 4: ĐIỂM "ĂN TIỀN" CỦA DỊCH VỤ LẤN CHIẾM (GRPC_DET_ENCROACH)

### 1. Polygon ROI linh hoạt theo chuẩn CMS
*Mã nguồn:* [`encroach_tracker.py`](file:///home/tado/long_dev/alarm/grpc_det_encroach/app/services/encroach_tracker.py)
* Không giới hạn ở hình chữ nhật cố định. Hệ thống đọc cấu hình đa giác tùy biến (Polygon) từ CMS UI qua hàm `parse_roi()`.
* Tự động nhận diện tọa độ chuẩn hóa ($0.0 - 1.0$) hoặc pixel tuyệt đối, kiểm tra vật thể lấn chiếm nằm trong hay ngoài đa giác bằng `cv2.pointPolygonTest`.

### 2. Wall-Clock Dwell Time Tracking (Chống sai lệch FPS mạng)
* **Lỗi kinh điển của các hệ thống non trẻ:** Đếm thời gian bằng số frame (ví dụ: camera 25 FPS $ightarrow$ 250 frame là 10 giây). Khi mạng lag, camera tụt xuống 5 FPS thì vật thể phải đỗ tới 50 giây mới bị bắt lỗi.
* **Giải pháp:** Sử dụng thời gian thực của hệ điều hành qua `time.monotonic()`:
  $$	ext{Dwell} = 	ext{now} - 	ext{track.roi\_since}$$
* Ngưỡng vi phạm $10	ext{s}$ (`DWELL_SECONDS = 10`) bảo đảm chính xác tuyệt đối trong mọi điều kiện mạng giật lag.

### 3. Quản lý vòng đời vết (Track Lifecycle Management)
* Tracker quản lý trạng thái độc lập cho từng camera: `roi_since` (thời điểm vào vùng cấm), `last_seen` (lần cuối nhìn thấy), `alarmed_at` (thời điểm đã bắn báo động).
* Bàn ghế, xe cộ rời khỏi vỉa hè $ightarrow$ tự động giải phóng track. Khi tái vi phạm $ightarrow$ kích hoạt chu kỳ dwell time mới.

---

## 🚀 PHẦN 5: TỐI ƯU HÓA PHẦN CỨNG & NẠP LUỒNG (CHUNG CHO CẢ 2 SERVICE)

### 1. gRPC Dynamic Batching trên TensorRT 10
*Mã nguồn:* [`server.py`](file:///home/tado/long_dev/alarm/grpc_det_weapon/app/server.py)
* **Sliding Window Batching:** Hàng đợi gom frame từ nhiều camera, mở cửa sổ $10	ext{ms}$ (`BATCH_TIMEOUT_MS`) để gom tối đa 8 frame (`BATCH_SIZE`) rồi nạp 1 lần vào GPU TensorRT. Tăng throughput lên gấp 3–4 lần so với chạy đơn lẻ.
* **Dead Request Cancellation:** Khi hệ thống chịu tải cao, nếu client đã hủy RPC hoặc quá hạn deadline (`ctx.is_active() == False`), worker loại bỏ frame ngay trước khi đưa vào GPU, không lãng phí tài nguyên tính toán.

### 2. Dual-Mode Stream Ingestion (Server thụ động & Worker chủ động)
*Mã nguồn:* [`stream_runner.py`](file:///home/tado/long_dev/alarm/grpc_det_weapon/app/services/stream_runner.py)
* Vừa đóng vai trò **gRPC Server** nhận frame từ gateway bên ngoài, vừa tích hợp **`StreamRunnerManager`** tự động kết nối và đọc camera.
* **Ép RTSP chạy qua TCP** (`OPENCV_FFMPEG_CAPTURE_OPTIONS = rtsp_transport;tcp`): Ngăn chặn triệt để hiện tượng rớt gói UDP gây vỡ hình, nhòe frame và sọc xanh trên chuẩn nén H.264/H.265.
* **Tích hợp C-SDK chính hãng:** Tích hợp trực tiếp Dahua NetSDK và Hikvision HCNetSDK qua C-bindings, cho phép lấy luồng video mượt mà khi camera nằm trong mạng nội bộ không mở port RTSP.
* **Khống chế FPS thông minh (`~2 FPS`):** Đối với bài toán an ninh giám sát vũ khí và lấn chiếm, tần số 2 FPS ($0.5	ext{s}$ một frame) là hoàn hảo để phản ứng kịp thời, đồng thời giảm tải GPU tới 90%, cho phép 1 GPU gánh hàng chục camera đồng thời.

---

## 🎤 PHẦN 6: BỘ CÂU HỎI MOCK INTERVIEW & ĐÁP ÁN MẪU

### ❓ Q1: "Nếu kẻ gian giấu vũ khí trong túi hoặc chỉ lộ cán dao thì Mask Overlap có hoạt động được không?"
> **Trả lời:**
> *"Hệ thống được thiết kế với 2 cơ chế tương hỗ:*
> *1. Mô hình YOLOv8-seg được huấn luyện trên tập dữ liệu đa dạng các góc cầm nắm (cán dao, báng súng).*
> *2. Thuật toán **Dilation** của em có thể cấu hình `DILATION_KERNEL_SIZE` từ 5 đến 15 pixel để mở rộng biên bao quanh bàn tay và cổ tay. Chỉ cần cán dao chạm vào vùng tay cầm và đạt ngưỡng tối thiểu 15 pixel giao thoa (`MIN_OVERLAP_PIXELS = 15`), thuật toán đã xác nhận và kích hoạt báo động ngay lập tức."*

### ❓ Q2: "Nếu camera bị mất mạng đột ngột thì Stream Runner xử lý ra sao?"
> **Trả lời:**
> *"Trong `stream_runner.py`, vòng lặp đọc camera được bọc trong khối quản lý trạng thái tự phục hồi (Self-healing). Khi mất frame, hệ thống gọi `cap.release()`, cập nhật trạng thái camera về offline (`patch_camera(cam_id, 0)`), ngủ 2–5 giây và tự động khởi tạo kết nối lại. Khi mạng phục hồi, stream tự động hoạt động trở lại mà không cần sự can thiệp của con người."*

### ❓ Q3: "Tại sao trong dịch vụ vũ khí bạn không lưu ảnh cục bộ mà đẩy qua ThreadPool lên CMS?"
> **Trả lời:**
> *"Để đảm bảo tính phi trạng thái (Stateless) của Microservice AI. Dịch vụ AI chỉ tập trung vào năng lực tính toán và phát hiện. Việc lưu trữ phân tán, phân quyền người dùng và phát thông báo realtime (SocketIO/Push Notification) thuộc về trách nhiệm của CMS Backend.*
> *Việc sử dụng `ThreadPoolExecutor(max_workers=4)` cho hàm `_post_alarm` giúp việc gửi HTTP POST diễn ra hoàn toàn bất đồng bộ, không làm gián đoạn vòng lặp nóng (hot loop) của luồng xử lý video."*

---

## 📋 BẢNG THAM SỐ CẤU HÌNH HỆ THỐNG ĐỂ TỰ TIN ĐỐI ĐÁP

| Tham số cấu hình | Giá trị | Ý nghĩa thực chiến |
|---|---|---|
| `BATCH_SIZE` | `8` | Kích thước batch tối đa cho Dynamic Batching GPU |
| `BATCH_TIMEOUT_MS`| `10 ms` | Cửa sổ gom frame vào batch suy luận |
| `STREAM_FPS_INTERVAL`| `0.5 s (~2 FPS)` | Giới hạn FPS đọc camera để tối ưu hóa năng lực GPU |
| `DILATION_KERNEL_SIZE`| `5 - 15` | Kernel làm nở mask người bù đắp vùng tiếp xúc tay cầm |
| `MIN_OVERLAP_PIXELS` | `15 px` | Số pixel tối thiểu chứng minh vũ khí đang được người cầm |
| `MIN_OVERLAP_RATIO` | `0.05 (5%)` | Tỷ lệ diện tích vũ khí tối thiểu phải giao thoa với người |
| `ALARM_COOLDOWN` | `10 s` | Thời gian làm nguội chống spam cảnh báo trên cùng camera |
| `DWELL_SECONDS` | `10 s` | Thời gian đỗ tối thiểu trong ROI để báo lỗi lấn chiếm |
| `RTSP_TRANSPORT` | `TCP` | Ép giao thức truyền dẫn RTSP qua TCP chống mất gói UDP |
| `POLL_INTERVAL` | `20 s` | Chu kỳ đồng bộ camera và license từ CMS Backend |
