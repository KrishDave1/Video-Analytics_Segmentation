"""
Refined YOLO -> SAM pipeline for automated phone segmentation.

Improvements over the original automatic_sam_gpt.py:
 - Load YOLO and SAM models once (not per image).
 - Run YOLO on the same resized image that SAM uses (coordinate consistency).
 - Use a combined selection metric (IoU between SAM mask and YOLO box + SAM score).
 - Optional merging of top-k masks when union improves IoU.
 - Optional RLE saving using pycocotools (if available).
 - Non-max suppression (NMS) on YOLO boxes to remove duplicates.
 - Configurable thresholds via CLI args and clearer logging.
 - Save COCO-style polygons and optional RLE for exact masks.
 - Better checks / warnings for empty masks or invalid crops.

Dependencies:
 - torch, torchvision
 - ultralytics (YOLOv8/YOLOv11 API compatibility)
 - segment-anything (SAM)
 - opencv-python
 - numpy
 - pycocotools (optional, recommended for RLE saving)
"""
from pathlib import Path
import argparse
import os
import json
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

try:
    from pycocotools import mask as mask_util  # optional
    HAS_PYCOCO = True
except Exception:
    HAS_PYCOCO = False

# -----------------------------
# Utilities
# -----------------------------
def resize_to_max_long_edge(image, max_long_edge):
    h, w = image.shape[:2]
    long_edge = max(h, w)
    if max_long_edge > 0 and long_edge > max_long_edge:
        scale = max_long_edge / long_edge
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return resized, scale
    return image, 1.0

def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return [int(x1), int(y1), int(round(x2 - x1)), int(round(y2 - y1))]

def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]

def bbox_iou_xyxy(a, b):
    # a and b are [x1,y1,x2,y2]
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union

def compute_mask_bbox(mask_bool):
    ys, xs = np.where(mask_bool)
    if xs.size == 0 or ys.size == 0:
        return [0, 0, 0, 0]
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    return [int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)]

def mask_to_polygons(mask_bool, epsilon=1.0):
    mask_uint8 = (mask_bool.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if c.shape[0] < 3:
            continue
        # optionally approximate
        approx = cv2.approxPolyDP(c, epsilon, True)
        poly = approx.reshape(-1, 2).astype(float).tolist()
        flat = [float(x) for p in poly for x in p]
        if len(flat) >= 6:
            polys.append(flat)
    return polys

def save_rle(mask_bool):
    # returns a serializable RLE dict if pycocotools is available
    if not HAS_PYCOCO:
        return None
    mask_uint8 = (mask_bool.astype(np.uint8))
    rle = mask_util.encode(np.asfortranarray(mask_uint8))
    # pycocotools returns counts as bytes in Python3; convert to list or string for json
    if isinstance(rle.get('counts'), bytes):
        rle['counts'] = rle['counts'].decode('ascii')
    return rle

def save_mask_png(mask_bool, path):
    mask_img = (mask_bool.astype(np.uint8)) * 255
    cv2.imwrite(path, mask_img)

def save_cropped_object(image_bgr, mask_bool, path):
    mask_uint8 = (mask_bool.astype(np.uint8)) * 255
    if mask_uint8.sum() == 0:
        return False
    x, y, w, h = compute_mask_bbox(mask_bool)
    if w == 0 or h == 0:
        return False
    masked = cv2.bitwise_and(image_bgr, image_bgr, mask=mask_uint8)
    crop = masked[y:y+h, x:x+w]
    cv2.imwrite(path, crop)
    return True

def nms_boxes(boxes, scores, iou_thresh=0.5):
    # boxes: list of [x1,y1,x2,y2], scores: list of float
    if len(boxes) == 0:
        return []
    boxes_arr = np.array(boxes, dtype=float)
    scores = np.array(scores, dtype=float)
    x1 = boxes_arr[:,0]; y1 = boxes_arr[:,1]; x2 = boxes_arr[:,2]; y2 = boxes_arr[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]
    return keep

# -----------------------------
# Selection logic
# -----------------------------
def select_best_mask_for_box(yolo_box_xyxy, sam_masks, sam_scores, iou_thresh=0.2, w_iou=0.7, w_score=0.3, try_merge_topk=2):
    """
    yolo_box_xyxy: [x1,y1,x2,y2] in same coords as sam_masks
    sam_masks: list/np.array of boolean masks (H,W)
    sam_scores: list/np.array of floats (same order)
    Returns: best_mask (bool array) or None
    """
    if len(sam_masks) == 0:
        return None, None

    # Compute bboxes and IoUs
    bboxes = []
    ious = []
    for m in sam_masks:
        bbox = compute_mask_bbox(m)
        bbox_xyxy = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
        bboxes.append(bbox)
        ious.append(bbox_iou_xyxy(yolo_box_xyxy, bbox_xyxy))

    ious = np.array(ious, dtype=float)
    scores = np.array(sam_scores, dtype=float)

    # normalize
    norm_iou = ious.copy()
    if norm_iou.max() > 0:
        norm_iou = norm_iou / (norm_iou.max() + 1e-12)
    norm_scores = scores.copy()
    if norm_scores.max() > 0:
        norm_scores = norm_scores / (norm_scores.max() + 1e-12)

    combined = w_iou * norm_iou + w_score * norm_scores
    best_idx = int(np.argmax(combined))
    if ious[best_idx] >= iou_thresh:
        return sam_masks[best_idx], float(scores[best_idx])

    # try merging top-k masks (e.g., top2) if allowed
    if try_merge_topk >= 2 and len(sam_masks) >= 2:
        topk_idx = combined.argsort()[::-1][:try_merge_topk]
        union_mask = np.zeros_like(sam_masks[0], dtype=bool)
        for idx in topk_idx:
            union_mask = np.logical_or(union_mask, sam_masks[int(idx)])
        # recompute IoU by area intersection with yolo box
        y1, x1, y2, x2 = int(yolo_box_xyxy[1]), int(yolo_box_xyxy[0]), int(yolo_box_xyxy[3]), int(yolo_box_xyxy[2])
        # bounding box of union mask
        bbox_union = compute_mask_bbox(union_mask)
        bbox_xyxy_union = [bbox_union[0], bbox_union[1], bbox_union[0]+bbox_union[2], bbox_union[1]+bbox_union[3]]
        iou_union = bbox_iou_xyxy(yolo_box_xyxy, bbox_xyxy_union)
        if iou_union >= iou_thresh:
            avg_score = float(scores[topk_idx].mean())
            return union_mask, avg_score

    # otherwise reject
    return None, None

# -----------------------------
# Main processing
# -----------------------------
def process_image(img_path: str, sam_predictor: SamPredictor, yolo_model: YOLO, out_dir: str,
                  label: str, max_long_edge: int, iou_weight: float, score_weight: float,
                  min_iou: float, merge_topk: int, nms_iou: float, save_rle_flag: bool, verbose: bool):
    os.makedirs(out_dir, exist_ok=True)
    image_bgr = cv2.imread(img_path)
    if image_bgr is None:
        if verbose:
            print(f"[WARN] Could not read {img_path}")
        return

    # Keep original for saving outputs
    orig_h, orig_w = image_bgr.shape[:2]

    # Resize first (both YOLO and SAM will operate on this resized image)
    resized_bgr, scale = resize_to_max_long_edge(image_bgr, max_long_edge)
    resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

    # Run YOLO on resized image (consistent coordinates)
    results = yolo_model(resized_bgr)
    yolo_boxes_xyxy = []
    yolo_scores = []
    for r in results:
        # handle multiple detections per result
        if hasattr(r, 'boxes') and r.boxes is not None:
            # prefer xyxy for clarity
            boxes_xyxy = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            for b, s in zip(boxes_xyxy, scores):
                yolo_boxes_xyxy.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
                yolo_scores.append(float(s))

    if len(yolo_boxes_xyxy) == 0:
        if verbose:
            print(f"[INFO] No YOLO detections in {img_path}")
        return

    # NMS on YOLO boxes to remove overlapping duplicates
    keep_idx = nms_boxes(yolo_boxes_xyxy, yolo_scores, iou_thresh=nms_iou)
    yolo_boxes_xyxy = [yolo_boxes_xyxy[i] for i in keep_idx]
    yolo_scores = [yolo_scores[i] for i in keep_idx]

    if verbose:
        print(f"[INFO] YOLO detected {len(yolo_boxes_xyxy)} boxes after NMS (scale={scale:.3f})")

    # Set image for SAM predictor (resized coords)
    sam_predictor.set_image(resized_rgb)

    per_image_dir = Path(out_dir) / Path(img_path).stem
    per_image_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for idx, (ybox, yscore) in enumerate(zip(yolo_boxes_xyxy, yolo_scores)):
        # ybox is in resized coords
        # call SAM predictor with box prompt
        box_np = np.array(ybox, dtype=float)[None, :]
        masks, scores, logits = sam_predictor.predict(box=box_np, multimask_output=True)
        # masks: (K, H_resized, W_resized)
        # choose best mask using selection rule
        best_mask_resized, best_score = select_best_mask_for_box(
            yolo_box_xyxy=ybox,
            sam_masks=[m.astype(bool) for m in masks],
            sam_scores=list(scores),
            iou_thresh=min_iou,
            w_iou=iou_weight,
            w_score=score_weight,
            try_merge_topk=merge_topk
        )

        if best_mask_resized is None:
            if verbose:
                print(f"[WARN] No suitable SAM mask for YOLO box #{idx} (score={yscore:.3f})")
            continue

        # Upscale mask to original size
        mask_uint8_small = (best_mask_resized.astype(np.uint8)) * 255
        orig_size = (orig_w, orig_h)
        mask_up = cv2.resize(mask_uint8_small, orig_size, interpolation=cv2.INTER_NEAREST).astype(bool)

        # Convert mask to bbox, polygons etc at original coords
        bbox = compute_mask_bbox(mask_up)  # [xmin,ymin,w,h]
        polygons = mask_to_polygons(mask_up)

        # Optionally save RLE
        rle = save_rle(mask_up) if save_rle_flag else None

        # Save mask PNG and crop
        base = Path(img_path).stem
        mask_path = per_image_dir / f"{base}_mask_{idx}.png"
        crop_path = per_image_dir / f"{base}_crop_{idx}.png"
        vis_path = per_image_dir / f"{base}_vis_{idx}.jpg"
        save_mask_png(mask_up, str(mask_path))
        saved_crop = save_cropped_object(image_bgr, mask_up, str(crop_path))

        # Visualization blended (overlay mask green and YOLO box in red)
        color_mask = np.zeros_like(image_bgr)
        color_mask[mask_up] = (0, 255, 0)
        blended = cv2.addWeighted(image_bgr, 0.7, color_mask, 0.3, 0)
        # draw box (scale YOLO box back to original coords)
        # ybox is in resized coords; scale up by 1/scale to original
        inv_scale = 1.0 / scale if scale != 0 else 1.0
        bx1, by1, bx2, by2 = [int(round(v * inv_scale)) for v in ybox]
        cv2.rectangle(blended, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
        cv2.imwrite(str(vis_path), blended)

        # Write JSON
        ann = {
            "image": str(Path(img_path).name),
            "mask_id": int(idx),
            "bbox": [int(b) for b in bbox],
            "area": int(mask_up.sum()),
            "sam_score": float(best_score),
            "yolo_score": float(yscore),
            "label": label,
            "polygons": polygons,
            "rle": rle,
            "size": {"height": orig_h, "width": orig_w}
        }
        json_path = per_image_dir / f"{base}_ann_{idx}.json"
        with open(json_path, "w") as f:
            json.dump(ann, f, indent=2)

        saved += 1
        if verbose:
            print(f"[OK] Saved mask #{idx}: {mask_path}, crop:{crop_path}, ann:{json_path}")

    if verbose:
        print(f"[INFO] Processed {img_path} => saved {saved} masks")


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Refined YOLO -> SAM automated segmentation")
    p.add_argument("--checkpoint", required=True, help="Path to SAM checkpoint (.pth)")
    p.add_argument("--model-type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"], help="SAM model type")
    p.add_argument("--yolo-model", required=True, help="Path to YOLO model (e.g., best.pt)")
    p.add_argument("--input", required=True, help="Input image or directory")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--label", default="mobile_phone", help="Label for annotations")
    p.add_argument("--max-long-edge", type=int, default=1024, help="Resize long edge before processing (0 disables)")
    p.add_argument("--iou-weight", type=float, default=0.7, help="Weight for IoU in mask selection (0..1)")
    p.add_argument("--score-weight", type=float, default=0.3, help="Weight for SAM score in mask selection (0..1)")
    p.add_argument("--min-iou", type=float, default=0.2, help="Minimum IoU between mask bbox and YOLO box to accept")
    p.add_argument("--merge-topk", type=int, default=2, help="Try merging top-k masks to improve IoU (1 disables)")
    p.add_argument("--nms-iou", type=float, default=0.5, help="NMS IoU threshold for YOLO boxes")
    p.add_argument("--save-rle", action="store_true", help="Save pycocotools RLE in JSON (requires pycocotools)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Device to run models")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    return p.parse_args()

def main():
    args = parse_args()
    device_requested = args.device
    if device_requested == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[INFO] Using device: {device}")

    # Load SAM once
    ckpt = args.checkpoint
    sam = sam_model_registry[args.model_type](checkpoint=ckpt)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # Load YOLO once
    yolo = YOLO(args.yolo_model)

    # Gather images
    inp = Path(args.input)
    if inp.is_dir():
        images = sorted(list(inp.glob("*.jpg")) + list(inp.glob("*.png")) + list(inp.glob("*.jpeg")))
    else:
        images = [inp]

    for img in images:
        process_image(
            str(img),
            sam_predictor=predictor,
            yolo_model=yolo,
            out_dir=args.output,
            label=args.label,
            max_long_edge=args.max_long_edge,
            iou_weight=args.iou_weight,
            score_weight=args.score_weight,
            min_iou=args.min_iou,
            merge_topk=args.merge_topk,
            nms_iou=args.nms_iou,
            save_rle_flag=args.save_rle,
            verbose=args.verbose
        )

    print("[DONE] All images processed.")

if __name__ == "__main__":
    main()