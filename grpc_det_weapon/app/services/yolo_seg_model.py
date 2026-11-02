"""
yolo_seg_model.py — Unified YOLO-seg inference module for weapon detector.

Khác biệt chính so với yolo_model.py (detection-only) của encroach:
  - Trả về cả binary mask segmentation cho mỗi đối tượng.
  - Hỗ trợ TensorRT engine dạng segmentation.

settings.py must define:
    MODEL_PT          path to .pt weights (fallback + export source)
    MODEL_ENGINE      path to TensorRT engine (auto-built on the running GPU)
    CLASS_NAMES       {class_id: name}
    CONF_THRESHOLD    float confidence threshold
    EXPORT_DYNAMIC_BATCH  True → TRT engine exported with dynamic batch
"""
import os

os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["CHECK_REQUIREMENTS"] = "false"

import shutil

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from core.settings import settings


def init_yolo():
    """
    Init YOLO-seg model with 3-layer strategy:
      1) GPU + USE_TENSORRT: load existing TRT engine, else export from .pt then load.
      2) Export/load failure → fall back to PyTorch .pt on GPU.
      3) No GPU → PyTorch .pt on CPU.
    Returns: (model, device, model_is_pt)
    """
    engine_path = settings.MODEL_ENGINE
    pt_path = settings.MODEL_PT
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[YOLO-SEG] device: {device} | USE_TENSORRT: {settings.USE_TENSORRT} | "
          f"engine: {engine_path}")

    net = None
    model_is_pt = False

    if device != "cpu" and settings.USE_TENSORRT:
        if os.path.isfile(engine_path):
            try:
                print("[YOLO-SEG] Loading TensorRT engine...")
                net = YOLO(engine_path)
                print("✅ Loaded TensorRT engine.")
            except Exception as e:
                print(f"⚠️  Failed to load engine: {e}")
        else:
            print("⚠️  No TensorRT engine file found.")

        if net is None and os.path.isfile(pt_path):
            try:
                dynamic = bool(getattr(settings, "EXPORT_DYNAMIC_BATCH", True))
                print(f"[YOLO-SEG] Exporting .pt → TensorRT engine "
                      f"(dynamic={dynamic}, batch=8)...")
                tmp = YOLO(pt_path)
                exported_path = tmp.export(format="engine", dynamic=dynamic, batch=8, workspace=1)
                print(f"[YOLO-SEG] Engine exported to: {exported_path}")

                if os.path.isfile(exported_path):
                    shutil.move(exported_path, engine_path)
                    print(f"[YOLO-SEG] Moved engine to: {engine_path}")

                net = YOLO(engine_path)
                print("✅ Exported & loaded new TensorRT engine.")
            except Exception as e:
                print(f"⚠️  Export/load engine failed: {e}")
            finally:
                if 'tmp' in locals():
                    del tmp
                if device != "cpu":
                    torch.cuda.empty_cache()

    if net is None:
        if not os.path.isfile(pt_path):
            raise FileNotFoundError(f"Không tìm thấy {pt_path}")
        print(f"[YOLO-SEG] Loading PyTorch .pt model from {pt_path}...")
        net = YOLO(pt_path)
        model_is_pt = True
        if device != "cpu":
            try:
                torch.cuda.empty_cache()
                net.to(device)
                print("✅ Loaded PyTorch .pt model on GPU.")
            except RuntimeError as oom_err:
                print(f"⚠️ GPU OOM khi tải PyTorch model ({oom_err}), fallback sang CPU.")
                torch.cuda.empty_cache()
                device = "cpu"
                net.to(device)
        else:
            print("✅ Loaded PyTorch .pt model on CPU.")

    return net, device, model_is_pt


def _parse_seg_results(results, item, conf_thresh):
    """
    Extract filtered + rescaled boxes AND binary masks from one ultralytics
    segmentation result.

    Returns list of dicts:
      {x1, y1, x2, y2, class_name, class_id, confidence, mask}
      mask: np.ndarray uint8 (orig_h, orig_w) — binary mask scaled to original frame size.
    """
    orig_w, orig_h = item["orig_w"], item["orig_h"]
    allowed_classes = item.get("allowed_classes")
    h_rs, w_rs = item["img"].shape[:2]
    scale_x = orig_w / w_rs
    scale_y = orig_h / h_rs

    class_names = getattr(settings, "CLASS_NAMES", {})

    output = []

    # Trích xuất masks — ultralytics trả masks.data tensor (N, mask_h, mask_w)
    has_masks = (results.masks is not None and results.masks.data is not None
                 and len(results.masks.data) > 0)

    for idx, box in enumerate(results.boxes):
        conf = float(box.conf[0].item() if isinstance(box.conf, torch.Tensor)
                     and box.conf.numel() > 0 else box.conf)
        cls = int(box.cls[0].item() if isinstance(box.cls, torch.Tensor)
                  and box.cls.numel() > 0 else box.cls)
        if conf < conf_thresh:
            continue
        if allowed_classes is not None and cls not in allowed_classes:
            continue

        xy = box.xyxy[0].tolist() if isinstance(box.xyxy, torch.Tensor) else box.xyxy[0]
        x1g = max(0, min(int(xy[0] * scale_x), orig_w - 1))
        y1g = max(0, min(int(xy[1] * scale_y), orig_h - 1))
        x2g = max(0, min(int(xy[2] * scale_x), orig_w - 1))
        y2g = max(0, min(int(xy[3] * scale_y), orig_h - 1))

        # Binary mask: resize từ kích thước model → kích thước ảnh gốc
        mask = None
        if has_masks and idx < len(results.masks.data):
            raw_mask = results.masks.data[idx]
            if isinstance(raw_mask, torch.Tensor):
                raw_mask = raw_mask.cpu().numpy()
            # raw_mask: (mask_h, mask_w) float 0..1 → resize → threshold → uint8
            mask_resized = cv2.resize(raw_mask.astype(np.float32),
                                      (orig_w, orig_h),
                                      interpolation=cv2.INTER_LINEAR)
            mask = (mask_resized > 0.5).astype(np.uint8)

        output.append({
            "x1": x1g, "y1": y1g, "x2": x2g, "y2": y2g,
            "class_name": class_names.get(cls, "other"),
            "class_id": cls,
            "confidence": conf,
            "mask": mask,
        })
    return output


def detect_frame_seg(model, device, model_is_pt, img_resized, orig_w, orig_h,
                     allowed_classes=None) -> list:
    """Detect + Segment on a single frame → list of box+mask dicts."""
    conf_thresh = float(settings.CONF_THRESHOLD)
    item = {"img": img_resized, "orig_w": orig_w, "orig_h": orig_h,
            "allowed_classes": allowed_classes}

    if model_is_pt:
        results = model(img_resized, device=device, verbose=False)[0]
    else:
        results = model.predict(source=img_resized, device=device, verbose=False)[0]
    return _parse_seg_results(results, item, conf_thresh)


def detect_batch_seg(model, device, model_is_pt, batch_items: list) -> list:
    """
    Batched segmentation for N frames in one GPU forward pass.
    Falls back to sequential per-frame if the TRT engine doesn't support batch>1.
    """
    if not batch_items:
        return []

    conf_thresh = float(settings.CONF_THRESHOLD)

    if len(batch_items) == 1:
        item = batch_items[0]
        return [detect_frame_seg(model, device, model_is_pt, item["img"],
                                 item["orig_w"], item["orig_h"], item["allowed_classes"])]

    imgs_list = [item["img"] for item in batch_items]

    try:
        if model_is_pt:
            results_list = model(imgs_list, device=device, verbose=False)
        else:
            results_list = model.predict(source=imgs_list, device=device, verbose=False)
    except Exception as exc:
        print(f"⚠️ [YOLO-SEG BATCH FALLBACK] Engine không hỗ trợ batch>{len(batch_items)} "
              f"({exc}). Chuyển sang xử lý tuần tự...")
        return [detect_frame_seg(model, device, model_is_pt, item["img"],
                                 item["orig_w"], item["orig_h"], item["allowed_classes"])
                for item in batch_items]

    return [_parse_seg_results(results, batch_items[idx], conf_thresh)
            for idx, results in enumerate(results_list)]
