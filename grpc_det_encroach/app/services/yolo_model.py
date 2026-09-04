"""
yolo_model.py — Unified YOLO inference module for ALL gRPC detector services.
(identical bytes in every detector; behaviour driven by core/settings.py)

settings.py must define:
    MODEL_PT          path to .pt weights (fallback + export source)
    MODEL_ENGINE      path to TensorRT engine (auto-built on the running GPU)
    CLASS_NAMES       {class_id: name}
    CONF_THRESHOLD    float confidence threshold
    EXPORT_DYNAMIC_BATCH  True → TRT engine exported with dynamic batch (multi-request batching)
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
    Init YOLO model with 3-layer strategy:
      1) GPU + USE_TENSORRT: load existing TRT engine, else export from .pt (on THIS GPU) then load.
      2) Export/load failure → fall back to PyTorch .pt on GPU.
      3) No GPU → PyTorch .pt on CPU.
    Returns: (model, device, model_is_pt)
    """
    engine_path = settings.MODEL_ENGINE
    pt_path = settings.MODEL_PT
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[YOLO] device: {device} | USE_TENSORRT: {settings.USE_TENSORRT} | "
          f"engine: {engine_path}")

    net = None
    model_is_pt = False

    if device != "cpu" and settings.USE_TENSORRT:
        if os.path.isfile(engine_path):
            try:
                print("[YOLO] Loading TensorRT engine...")
                net = YOLO(engine_path)
                print("✅ Loaded TensorRT engine.")
            except Exception as e:
                print(f"⚠️  Failed to load engine: {e}")
        else:
            print("⚠️  No TensorRT engine file found.")

        if net is None and os.path.isfile(pt_path):
            try:
                dynamic = bool(getattr(settings, "EXPORT_DYNAMIC_BATCH", True))
                print(f"[YOLO] Exporting .pt → TensorRT engine "
                      f"(dynamic={dynamic}, batch=8)...")
                tmp = YOLO(pt_path)
                exported_path = tmp.export(format="engine", dynamic=dynamic, batch=8)
                print(f"[YOLO] Engine exported to: {exported_path}")

                if os.path.isfile(exported_path):
                    shutil.move(exported_path, engine_path)
                    print(f"[YOLO] Moved engine to: {engine_path}")

                net = YOLO(engine_path)
                print("✅ Exported & loaded new TensorRT engine.")
            except Exception as e:
                print(f"⚠️  Export/load engine failed: {e}")

    if net is None:
        if not os.path.isfile(pt_path):
            raise FileNotFoundError(f"Không tìm thấy {pt_path}")
        print(f"[YOLO] Loading PyTorch .pt model from {pt_path}...")
        net = YOLO(pt_path)
        model_is_pt = True
        if device != "cpu":
            net.to(device)
        print("✅ Loaded PyTorch .pt model.")

    return net, device, model_is_pt


def _parse_boxes(results, item, conf_thresh):
    """Extract filtered + rescaled boxes from one ultralytics result."""
    orig_w, orig_h = item["orig_w"], item["orig_h"]
    allowed_classes = item.get("allowed_classes")
    h_rs, w_rs = item["img"].shape[:2]
    scale_x = orig_w / w_rs
    scale_y = orig_h / h_rs

    class_names = getattr(settings, "CLASS_NAMES", {})

    output = []
    for box in results.boxes:
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

        output.append({
            "x1": x1g, "y1": y1g, "x2": x2g, "y2": y2g,
            "class_name": class_names.get(cls, "other"),
            "class_id": cls,
            "confidence": conf,
        })
    return output


def detect_frame(model, device, model_is_pt, img_resized, orig_w, orig_h,
                 allowed_classes=None) -> list:
    """Detect on a single frame → list of box dicts."""
    conf_thresh = float(settings.CONF_THRESHOLD)
    item = {"img": img_resized, "orig_w": orig_w, "orig_h": orig_h,
            "allowed_classes": allowed_classes}

    if model_is_pt:
        results = model(img_resized, device=device, verbose=False)[0]
    else:
        results = model.predict(source=img_resized, device=device, verbose=False)[0]
    return _parse_boxes(results, item, conf_thresh)


def detect_batch(model, device, model_is_pt, batch_items: list) -> list:
    """
    Batched detection for N frames in one GPU forward pass.
    Falls back to sequential per-frame if the TRT engine doesn't support batch>1.
    """
    if not batch_items:
        return []

    conf_thresh = float(settings.CONF_THRESHOLD)

    if len(batch_items) == 1:
        item = batch_items[0]
        return [detect_frame(model, device, model_is_pt, item["img"],
                             item["orig_w"], item["orig_h"], item["allowed_classes"])]

    imgs_list = [item["img"] for item in batch_items]

    try:
        if model_is_pt:
            results_list = model(imgs_list, device=device, verbose=False)
        else:
            results_list = model.predict(source=imgs_list, device=device, verbose=False)
    except Exception as exc:
        print(f"⚠️ [YOLO BATCH FALLBACK] Engine không hỗ trợ batch>{len(batch_items)} "
              f"({exc}). Chuyển sang xử lý tuần tự...")
        return [detect_frame(model, device, model_is_pt, item["img"],
                             item["orig_w"], item["orig_h"], item["allowed_classes"])
                for item in batch_items]

    return [_parse_boxes(results, batch_items[idx], conf_thresh)
            for idx, results in enumerate(results_list)]
