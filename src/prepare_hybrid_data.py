import os
import glob
import cv2
import numpy as np
import sys
from ultralytics import YOLO
from tqdm import tqdm

def prepare_hybrid_dataset(src_dir, dst_dir, yolo_model_path):
    os.makedirs(dst_dir, exist_ok=True)
    
    print(f"Loading YOLO model from {yolo_model_path}")
    model = YOLO(yolo_model_path)
    
    all_files = glob.glob(os.path.join(src_dir, "*.jpg"))
    valid_files = [f for f in all_files if ' ' in os.path.basename(f)]
    
    print(f"Processing {len(valid_files)} images...")
    
    for file_path in tqdm(valid_files):
        filename = os.path.basename(file_path)
        out_path = os.path.join(dst_dir, filename)
        
        # Read image
        img = cv2.imread(file_path)
        if img is None:
            continue
            
        # YOLO inference
        results = model(img, verbose=False)
        result = results[0]
        
        # Create empty mask
        h, w = img.shape[:2]
        master_mask = np.zeros((h, w), dtype=np.uint8)
        
        if result.masks is not None:
            # Ultralytics masks are resized back to original shape by default in recent versions,
            # but let's be safe and use xyn or similar if needed.
            # actually result.masks.data has shape (N, H, W)
            # but H,W might be 640x640, so let's use the normalized coordinates (xyn)
            for segments in result.masks.xyn:
                if len(segments) > 0:
                    # segments is Nx2, normalized
                    contour = (segments * np.array([w, h])).astype(np.int32)
                    cv2.fillPoly(master_mask, [contour], 255)
                    
        # Apply mask
        # We can also add a small bit of the original image (e.g. 10%) so it doesn't look completely alien,
        # but pure black background is usually better for CNN to ignore it.
        masked_img = cv2.bitwise_and(img, img, mask=master_mask)
        
        # Save
        cv2.imwrite(out_path, masked_img)

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SRC_DIR = os.path.join(BASE_DIR, "data", "dataset ++")
    DST_DIR = os.path.join(BASE_DIR, "data", "hybrid_dataset")
    
    # Path to 008 YOLO model
    YOLO_PATH = os.path.join(BASE_DIR, "..", "Dental_008", "runs", "segment", "yolo_dentex", "yolov8m_seg_finetune", "weights", "best.pt")
        
    prepare_hybrid_dataset(SRC_DIR, DST_DIR, YOLO_PATH)
