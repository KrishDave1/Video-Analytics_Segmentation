import json
import os
import numpy as np
import shutil
import cv2
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

phone_ann_dir = os.path.join(BASE_DIR, "annotations_sam_yolo") #Add annotations folder
archive_pos = os.path.join(BASE_DIR, "..", "archive", "positive")  # archive folder is basically the MUID-IITR dataset
archive_neg = os.path.join(BASE_DIR, "..", "archive", "negative")
global_img_dir = os.path.join(BASE_DIR, "images")
image_dir = os.path.join(BASE_DIR, "images")
annotated_dir = os.path.join(BASE_DIR, "annotated")

def load_phone_centroid(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    x, y, w, h = data["bbox"]
    cx = x + w / 2
    cy = y + h / 2
    return np.array([cx, cy])

def process_phone_annotations(
    phone_ann_dir,
    archive_pos,
    archive_neg,
    global_img_dir
):
    os.makedirs(global_img_dir, exist_ok=True)
    results = {}
    for folder in os.listdir(phone_ann_dir):
        folder_path = os.path.join(phone_ann_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        centroids = []
        for f in os.listdir(folder_path):
            if f.endswith(".json"):
                json_path = os.path.join(folder_path, f)
                centroid = load_phone_centroid(json_path)
                centroids.append(centroid)
        if len(centroids) == 0:
            continue
        results[folder] = centroids
        filename = folder + ".jpg"
        pos_file = os.path.join(archive_pos, filename)
        neg_file = os.path.join(archive_neg, filename)
        source = None
        if os.path.isfile(pos_file):
            source = pos_file
        elif os.path.isfile(neg_file):
            source = neg_file
        else:
            continue
        dest_path = os.path.join(global_img_dir, filename)
        shutil.copy(source, dest_path)
    return results

def get_all_face_centroids_yolo(image_path, model):
    img = cv2.imread(image_path)
    if img is None:
        return [], []
    results = model(img)
    face_centroids = []
    face_boxes = []
    for box in results[0].boxes:
        cls = int(box.cls[0])
        if cls != 0:
            continue
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        face_boxes.append((x1, y1, x2, y2))
        face_top = y1
        face_bottom = y1 + 0.30 * (y2 - y1)
        cx = (x1 + x2) / 2
        cy = (face_top + face_bottom) / 2
        face_centroids.append(np.array([cx, cy]))
    return face_centroids, face_boxes

def estimate_scale_from_face_box(x1, y1, x2, y2, real_face_width_cm=15.0):
    face_width_px = abs(x2 - x1)
    if face_width_px < 1:
        return None
    return real_face_width_cm / face_width_px

def draw_annotations(img, phones, faces, face_boxes, matches, save_path):
    for i, box in enumerate(face_boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        fcx, fcy = faces[i]
        cv2.circle(img, (int(fcx), int(fcy)), 8, (0,255,0), -1)
        cv2.putText(img, f"Face {i+1}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0),2)
    for i, p in enumerate(phones):
        px, py = p
        cv2.circle(img, (int(px), int(py)), 10, (255,0,0), -1)
        cv2.putText(img, f"Phone {i+1}", (int(px)+10, int(py)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0),2)
    for phone_idx, face_idx, used, dist_cm in matches:
        p = phones[phone_idx]
        f = faces[face_idx]
        color = (0,0,255) if used else (0,255,255)
        cv2.arrowedLine(img, (int(p[0]), int(p[1])), (int(f[0]), int(f[1])), color, 3, tipLength=0.1)
        cv2.putText(img, f"{dist_cm:.1f}cm", (int((p[0]+f[0])/2), int((p[1]+f[1])/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imwrite(save_path, img)

def match_and_detect_use(phone_centroids, image_dir, model, annotated_dir):
    os.makedirs(annotated_dir, exist_ok=True)
    for img_id, phones in phone_centroids.items():
        img_path = os.path.join(image_dir, img_id + ".jpg")
        img = cv2.imread(img_path)
        if img is None:
            continue
        faces, face_boxes = get_all_face_centroids_yolo(img_path, model)
        if len(faces) == 0:
            continue
        matches = []
        for p_idx, p in enumerate(phones):
            min_dist = 1e9
            best_face_idx = None
            for f_idx, f in enumerate(faces):
                dist = np.linalg.norm(p - f)
                if dist < min_dist:
                    min_dist = dist
                    best_face_idx = f_idx
            x1, y1, x2, y2 = face_boxes[best_face_idx]
            cm_per_px = estimate_scale_from_face_box(x1, y1, x2, y2)
            dist_cm = min_dist * cm_per_px
            used = dist_cm < 40
            matches.append((p_idx, best_face_idx, used, dist_cm))
        save_path = os.path.join(annotated_dir, img_id + "_annotated.jpg")
        draw_annotations(img, phones, faces, face_boxes, matches, save_path)

if __name__ == "__main__":
    phone_centroids_1 = process_phone_annotations(
        phone_ann_dir=phone_ann_dir,
        archive_pos=archive_pos,
        archive_neg=archive_neg,
        global_img_dir=global_img_dir
    )
    phone_centroids = phone_centroids_1.copy()
    face_model = YOLO("yolo11m")
    match_and_detect_use(
        phone_centroids=phone_centroids,
        image_dir=image_dir,
        model=face_model,
        annotated_dir=annotated_dir
    )
