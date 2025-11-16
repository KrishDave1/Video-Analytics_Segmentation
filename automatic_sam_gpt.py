import os
import cv2
import torch
import numpy as np
import json
from pathlib import Path
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

# -----------------------------
# Utility functions
# -----------------------------

def resize_to_max_long_edge(image, max_long_edge):
    h, w = image.shape[:2]
    long_edge = max(h, w)
    if long_edge > max_long_edge:
        scale = max_long_edge / long_edge
        new_size = (int(w * scale), int(h * scale))
        resized = cv2.resize(image, new_size)
        return resized, scale
    return image, 1.0

def detect_mobile_phones_yolo(yolo_model_path, image_bgr):
    model = YOLO(yolo_model_path)
    results = model(image_bgr)
    boxes = []
    for result in results:
        for box in result.boxes.xywh.cpu().numpy():
            boxes.append(box.astype(int))
    return boxes

def save_mask_png(mask, path):
    mask_img = (mask.astype(np.uint8)) * 255
    cv2.imwrite(path, mask_img)

def save_cropped_object(image_bgr, mask, path):
    mask_uint8 = mask.astype(np.uint8)
    if mask_uint8.sum() == 0:
        print(f"[WARN] Empty mask — skipping save for {path}")
        return

    masked = cv2.bitwise_and(image_bgr, image_bgr, mask=mask_uint8)
    x, y, w, h = cv2.boundingRect(mask_uint8)
    if w == 0 or h == 0:
        print(f"[WARN] Invalid crop region — skipping {path}")
        return

    crop = masked[y:y + h, x:x + w]
    cv2.imwrite(path, crop)

def mask_to_polygons(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        polygons.append(contour.reshape(-1, 2).tolist())
    return polygons

def write_annotation_json(json_path, image_path, mask_id, mask_info, label, size_hw):
    data = {
        "image": os.path.basename(image_path),
        "mask_id": mask_id,
        "bbox": mask_info["bbox"],
        "area": mask_info["area"],
        "predicted_iou": mask_info["predicted_iou"],
        "stability_score": mask_info["stability_score"],
        "label": label if label else "object",
        "polygons": mask_info["polygons"],
        "size": {"height": size_hw[0], "width": size_hw[1]}
    }
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

# -----------------------------
# Core SAM + YOLO pipeline
# -----------------------------
def process_image(image_path, sam_model, out_dir, label, max_long_edge):
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Reading image: {image_path}")
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"[WARN] Could not read {image_path}")
        return

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized_rgb, scale = resize_to_max_long_edge(image_rgb, max_long_edge)

    # Step 1: YOLO detection
    yolo_model_path = "best.pt"  # your trained YOLO model path
    yolo_boxes = detect_mobile_phones_yolo(yolo_model_path, image_bgr)
    if not yolo_boxes:
        print(f"[WARN] No mobile phones detected in {image_path}")
        return
    print(f"[INFO] YOLO detected {len(yolo_boxes)} object(s)")

    # Step 2: SAM guided segmentation
    predictor = SamPredictor(sam_model)
    predictor.set_image(resized_rgb)

    masks = []
    for i, box in enumerate(yolo_boxes):
        box_scaled = [int(v * scale) for v in box]
        x, y, w, h = box_scaled
        box_xyxy = np.array([x, y, x + w, y + h])

        masks_i, scores, logits = predictor.predict(
            box=box_xyxy[None, :],
            multimask_output=True
        )
        best_id = np.argmax(scores)
        best_mask_resized = masks_i[best_id]

        # --- resize mask back to original image size ---
        best_mask = cv2.resize(best_mask_resized.astype(np.uint8),
                               (image_rgb.shape[1], image_rgb.shape[0]),
                               interpolation=cv2.INTER_NEAREST).astype(bool)

        # ✅ Option 2: Keep only the mask inside YOLO bounding box
        y1, y2 = max(0, y), min(image_rgb.shape[0], y + h)
        x1, x2 = max(0, x), min(image_rgb.shape[1], x + w)
        mask_box = np.zeros_like(best_mask)
        mask_box[y1:y2, x1:x2] = best_mask[y1:y2, x1:x2]
        best_mask = mask_box

        masks.append({
            "bbox": [x, y, w, h],
            "segmentation": best_mask,
            "score": float(scores[best_id])
        })

        # Visualization: overlay green mask + red YOLO box
        color_mask = np.zeros_like(image_rgb)
        color_mask[best_mask] = (0, 255, 0)
        blended = cv2.addWeighted(image_bgr, 0.7, cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR), 0.3, 0)
        cv2.rectangle(blended, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imwrite(os.path.join(out_dir, f"{Path(image_path).stem}_mask_{i}.jpg"), blended)

    print(f"[OK] Generated {len(masks)} SAM masks guided by YOLO boxes")

    # Step 3: Save masks, crops, and JSON
    base = Path(image_path).stem
    per_image_dir = os.path.join(out_dir, base)
    os.makedirs(per_image_dir, exist_ok=True)

    for idx, m in enumerate(masks):
        mask_bool = m["segmentation"].astype(bool)
        mask_path = os.path.join(per_image_dir, f"{base}_mask_{idx}.png")
        crop_path = os.path.join(per_image_dir, f"{base}_crop_{idx}.png")
        save_mask_png(mask_bool, mask_path)
        save_cropped_object(image_bgr, mask_bool, crop_path)
        polygons = mask_to_polygons(mask_bool)
        json_path = os.path.join(per_image_dir, f"{base}_ann_{idx}.json")
        h, w = image_bgr.shape[:2]
        write_annotation_json(json_path, image_path, idx, {
            "bbox": m["bbox"],
            "area": int(mask_bool.sum()),
            "predicted_iou": m["score"],
            "stability_score": 1.0,
            "polygons": polygons
        }, label, (h, w))

    print(f"[OK] Saved all masks, crops, and JSON annotations for {image_path}")


# -----------------------------
# Main CLI
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLO + SAM Segmentation Pipeline")
    parser.add_argument("--checkpoint", required=True, help="Path to SAM checkpoint (.pth)")
    parser.add_argument("--model-type", default="vit_h", help="SAM model type: vit_h, vit_l, or vit_b")
    parser.add_argument("--input", required=True, help="Input image or directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--label", default="object", help="Optional label for JSON output")
    parser.add_argument("--max-long-edge", type=int, default=1024, help="Resize long edge to this before processing")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=device)

    input_path = Path(args.input)
    if input_path.is_dir():
        images = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
    else:
        images = [input_path]

    for img_path in images:
        process_image(str(img_path), sam, args.output, args.label, args.max_long_edge)

    print("[DONE] All images processed successfully.")
