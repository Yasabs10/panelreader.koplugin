import json
import torch
import numpy as np
import os
import sys
import argparse
import time
import signal
import cv2
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

def get_model():
    """Downloads model if not present locally, then returns YOLO instance."""
    model_name = "v2023.12.07_l_yv11/model.pt"
    local_dir = "./models" # You can change this to any persistent directory
    local_path = os.path.join(local_dir, model_name)

    if not os.path.exists(local_path):
        print("Model not found locally. Downloading from Hugging Face...")
        os.makedirs(local_dir, exist_ok=True)
        # This downloads to local_dir and returns the path
        path = hf_hub_download(
            repo_id="deepghs/manga109_yolo", 
            filename=model_name,
            local_dir=local_dir
        )
    else:
        print(f"Loading model from local path: {local_path}")
        path = local_path

    return YOLO(path).to("cuda" if torch.cuda.is_available() else "cpu")

def detect_gutters_and_refine_boxes(boxes, gray_image):
    """
    Use Hough Transform to detect white gutters between panels and refine box boundaries.
    boxes: List of [x1, y1, x2, y2]
    gray_image: Full grayscale image
    """
    if len(boxes) <= 1:
        return boxes
    
    refined_boxes = []
    height, width = gray_image.shape
    
    # Create binary image for gutter detection (white areas)
    _, gutter_mask = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY)
    
    # Detect vertical lines (gutters between columns)
    vertical_lines = cv2.HoughLinesP(
        255 - gutter_mask,  # Invert to detect white lines as dark lines
        rho=1,
        theta=np.pi/180,
        threshold=50,  # Minimum votes
        minLineLength=max(width // 10, 50),
        maxLineGap=10
    )
    
    # Detect horizontal lines (gutters between rows)
    horizontal_lines = cv2.HoughLinesP(
        255 - gutter_mask,
        rho=1,
        theta=np.pi/180,
        threshold=50,
        minLineLength=max(height // 10, 50),
        maxLineGap=10
    )
    
    # Group lines by position
    vertical_gutters = []
    horizontal_gutters = []
    
    if vertical_lines is not None:
        for line in vertical_lines:
            x1, y1, x2, y2 = line[0]
            # Check if it's mostly vertical
            if abs(x2 - x1) < 20:  # Nearly vertical
                avg_x = (x1 + x2) / 2
                vertical_gutters.append(avg_x)
    
    if horizontal_lines is not None:
        for line in horizontal_lines:
            x1, y1, x2, y2 = line[0]
            # Check if it's mostly horizontal
            if abs(y2 - y1) < 20:  # Nearly horizontal
                avg_y = (y1 + y2) / 2
                horizontal_gutters.append(avg_y)
    
    # Cluster similar gutter positions
    vertical_gutters = sorted(list(set(int(g) for g in vertical_gutters)))
    horizontal_gutters = sorted(list(set(int(g) for g in horizontal_gutters)))
    
    # Refine each box using detected gutters
    for box in boxes:
        x1, y1, x2, y2 = box
        refined_box = box.copy()
        
        # Find nearest left gutter
        left_gutters = [g for g in vertical_gutters if g < x1 and abs(g - x1) < width // 20]
        if left_gutters:
            refined_box[0] = max(left_gutters[-1] + 2, refined_box[0])  # Move right to gutter
        
        # Find nearest right gutter
        right_gutters = [g for g in vertical_gutters if g > x2 and abs(g - x2) < width // 20]
        if right_gutters:
            refined_box[2] = min(right_gutters[0] - 2, refined_box[2])  # Move left to gutter
        
        # Find nearest top gutter
        top_gutters = [g for g in horizontal_gutters if g < y1 and abs(g - y1) < height // 20]
        if top_gutters:
            refined_box[1] = max(top_gutters[-1] + 2, refined_box[1])  # Move down to gutter
        
        # Find nearest bottom gutter
        bottom_gutters = [g for g in horizontal_gutters if g > y2 and abs(g - y2) < height // 20]
        if bottom_gutters:
            refined_box[3] = min(bottom_gutters[0] - 2, refined_box[3])  # Move up to gutter
        
        refined_boxes.append(refined_box)
    
    return refined_boxes

def merge_boxes(box1, box2):
    """Returns a new box that encompasses both input boxes."""
    return [
        min(box1[0], box2[0]), # Smallest X1
        min(box1[1], box2[1]), # Smallest Y1
        max(box1[2], box2[2]), # Largest X2
        max(box1[3], box2[3])  # Largest Y3
    ]

def calculate_iou(box1, box2):
    """Calculate Intersection over Union of two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def check_containment(box1, box2, threshold=0.9):
    """Check if one box is contained within another by threshold percentage."""
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return False
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Check if intersection is at least threshold% of either box
    return (intersection / area1 >= threshold) or (intersection / area2 >= threshold)

def merge_overlapping_boxes(boxes, overlap_threshold=0.3):
    """
    Merge boxes that overlap by more than threshold and are on same Y-axis.
    boxes: List of [x1, y1, x2, y2]
    overlap_threshold: Minimum overlap ratio to merge (0.3 = 30%)
    """
    if len(boxes) <= 1:
        return boxes
    
    boxes = boxes.copy()
    merged = True
    
    while merged:
        merged = False
        i = 0
        
        while i < len(boxes):
            j = i + 1
            while j < len(boxes):
                box1, box2 = boxes[i], boxes[j]
                
                # Check containment first - if one box is 90% inside another, merge regardless
                if check_containment(box1, box2):
                    boxes[i] = merge_boxes(box1, box2)
                    boxes.pop(j)
                    merged = True
                    break
                
                # Check if boxes are on same Y-axis (row)
                y_overlap = min(box1[3], box2[3]) - max(box1[1], box2[1])
                min_height = min(box1[3] - box1[1], box2[3] - box2[1])
                same_row = y_overlap > min_height * 0.5  # 50% Y overlap
                
                if same_row:
                    # Calculate overlap area
                    x_overlap = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
                    overlap_area = x_overlap * y_overlap
                    
                    # Calculate individual areas
                    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                    
                    # Check overlap ratio
                    overlap_ratio = overlap_area / min(area1, area2)
                    
                    if overlap_ratio > overlap_threshold:
                        # Merge boxes
                        boxes[i] = merge_boxes(box1, box2)
                        boxes.pop(j)
                        merged = True
                        break
                j += 1
            if merged:
                break
            i += 1
    
    return boxes

def detect_rows_with_histogram(boxes):
    """
    Detect rows using Y-projection histogram to find significant gaps.
    boxes: List of [x1, y1, x2, y2]
    """
    if len(boxes) <= 1:
        return [boxes]
    
    # Get Y coordinates and create histogram
    y_coords = []
    for box in boxes:
        y_coords.extend([box[1], box[3]])  # Add both top and bottom Y coordinates
    
    min_y, max_y = min(y_coords), max(y_coords)
    histogram_height = max_y - min_y
    
    if histogram_height <= 0:
        return [boxes]
    
    # Create Y-projection histogram (count boxes at each Y level)
    hist_bins = 100  # Adjustable resolution
    histogram = np.zeros(hist_bins)
    
    for box in boxes:
        # Add contribution for each box's vertical span
        y_start = int((box[1] - min_y) / histogram_height * (hist_bins - 1))
        y_end = int((box[3] - min_y) / histogram_height * (hist_bins - 1))
        y_start = max(0, min(y_start, hist_bins - 1))
        y_end = max(0, min(y_end, hist_bins - 1))
        
        histogram[y_start:y_end + 1] += 1
    
    # Find significant gaps (local minima)
    gaps = []
    for i in range(1, hist_bins - 1):
        if histogram[i] < histogram[i-1] and histogram[i] < histogram[i+1]:
            # This is a local minimum - potential gap
            if histogram[i] < np.max(histogram) * 0.3:  # Threshold for "significant" gap
                gap_y = min_y + (i / hist_bins) * histogram_height
                gaps.append(gap_y)
    
    # Sort boxes by Y coordinate
    boxes_sorted = sorted(boxes, key=lambda b: b[1])
    
    # Group boxes into rows based on gaps
    rows = []
    current_row = [boxes_sorted[0]]
    
    for box in boxes_sorted[1:]:
        # Check if this box should start a new row based on gaps
        box_center_y = (box[1] + box[3]) / 2
        
        should_new_row = False
        for gap_y in gaps:
            if box_center_y > gap_y and current_row[-1][3] <= gap_y:
                should_new_row = True
                break
        
        # Fallback to original logic if no clear gap found
        if not should_new_row and box[1] > current_row[-1][3] * 0.95:
            should_new_row = True
        
        if should_new_row:
            rows.append(current_row)
            current_row = [box]
        else:
            current_row.append(box)
    
    rows.append(current_row)
    return rows

def xy_cut_sort(boxes, rtl=False):
    """
    Recursive XY-Cut sorting for comic panels using histogram-based row detection.
    boxes: List of [x1, y1, x2, y2]
    rtl: True for Manga, False for Western comics
    """
    if len(boxes) <= 1:
        return boxes

    # 1. Detect rows using Y-projection histogram
    rows = detect_rows_with_histogram(boxes)

    # 2. Sort each row horizontally
    sorted_panels = []
    for row in rows:
        # Sort each row by X1 (Western = Left to Right, Manga = Right to Left)
        row.sort(key=lambda b: b[0], reverse=rtl)
        sorted_panels.extend(row)
        
    return sorted_panels

# --- Execution ---
def main():
    parser = argparse.ArgumentParser(description="Detect and order comic panels")
    parser.add_argument('-i', '--input', required=True, help='Input image path')
    parser.add_argument('--timeout', type=int, default=300, help='Processing timeout in seconds')
    args = parser.parse_args()
    
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Processing timed out after {args.timeout} seconds")
    
    # Set timeout signal
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(args.timeout)
    
    try:
        img_path = args.input
        if not os.path.exists(img_path):
            print(f"Error: Image path not found: {img_path}")
            sys.exit(1)

        print("Loading mosesb model...")
        start_time = time.time()
        model = get_model()
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")

        print(f"Processing image: {img_path}")
        print(f"Image size: {os.path.getsize(img_path)} bytes")
        
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Process with timing
        start_time = time.time()
        results = model(img_path, imgsz=640, conf=0.25, iou=0.6)[0]
        inference_time = time.time() - start_time
        print(f"Inference completed in {inference_time:.2f} seconds")

        # 2. Extract raw boxes
        raw_boxes = results.boxes.xyxy.cpu().numpy().tolist()
        print(f"Found {len(raw_boxes)} raw boxes")

        if len(raw_boxes) == 0:
            print("No boxes detected. Try lowering the confidence threshold.")
            # Save empty result
            output = {"reading_order": []}
            with open("reading_order.json", "w") as f:
                json.dump(output, f, indent=4)
            print("Saved empty result to reading_order.json")
            return

        # 3. Merge overlapping boxes on same row
        # Adjust overlap_threshold (0.3 = 30%) as needed
        start_time = time.time()
        cleaned_boxes = merge_overlapping_boxes(raw_boxes, overlap_threshold=0.3)
        merge_time = time.time() - start_time
        print(f"Merged boxes in {merge_time:.2f} seconds: {len(cleaned_boxes)} boxes")

        # 4. Apply Kumiko sorting to the cleaned boxes
        start_time = time.time()
        ordered_boxes = xy_cut_sort(cleaned_boxes, rtl=True)
        sort_time = time.time() - start_time
        print(f"Sorted boxes in {sort_time:.2f} seconds")

        # Optional: Shrink-wrap the final boxes to the actual ink inside them
        print("Shrink-wrapping panels to actual content...")
        start_time = time.time()
        
        # Load the full image for shrink-wrapping
        full_img = cv2.imread(img_path)
        full_img_gray = cv2.cvtColor(full_img, cv2.COLOR_BGR2GRAY)
        
        final_output_boxes = []
        for box in ordered_boxes:
            x1, y1, x2, y2 = [int(c) for c in box]
            # Crop to the detected panel
            panel_roi = full_img_gray[y1:y2, x1:x2]
            # Find the ink (anything not white)
            _, ink_mask = cv2.threshold(panel_roi, 245, 255, cv2.THRESH_BINARY_INV)
            
            # Apply morphological closing to connect fragmented ink lines
            # Create a kernel for morphological operations (adjust size based on image resolution)
            kernel_size = max(3, min((x2 - x1) // 50, (y2 - y1) // 50, 7))  # Adaptive kernel size
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            
            # Closing: Dilation followed by Erosion
            ink_mask_closed = cv2.morphologyEx(ink_mask, cv2.MORPH_CLOSE, kernel)
            
            ink_coords = cv2.findNonZero(ink_mask_closed)
            
            if ink_coords is not None:
                ix, iy, iw, ih = cv2.boundingRect(ink_coords)
                final_output_boxes.append([x1 + ix, y1 + iy, x1 + ix + iw, y1 + iy + ih])
            else:
                final_output_boxes.append([x1, y1, x2, y2])
        
        shrink_time = time.time() - start_time
        print(f"Shrink-wrapped {len(final_output_boxes)} panels in {shrink_time:.2f} seconds")

        # 5. Refine panel boundaries using gutter detection
        print("Detecting gutters and refining panel boundaries...")
        start_time = time.time()
        refined_boxes = detect_gutters_and_refine_boxes(final_output_boxes, full_img_gray)
        gutter_time = time.time() - start_time
        print(f"Refined panel boundaries using gutters in {gutter_time:.2f} seconds")

        # Save result with the correct reading order and refined boxes
        output = {"reading_order": []}
        for i, box in enumerate(refined_boxes):
            output["reading_order"].append({
                "index": i + 1,
                "bbox": [int(c) for c in box]
            })

        with open("reading_order.json", "w") as f:
            json.dump(output, f, indent=4)

        total_time = load_time + inference_time + merge_time + sort_time + shrink_time + gutter_time
        print(f"✅ Detected {len(refined_boxes)} panels in correct reading order.")
        print(f"Total processing time: {total_time:.2f} seconds")
        
        # Cancel timeout
        signal.alarm(0)
        
    except TimeoutError as e:
        print(f"❌ {e}")
        print("Try:")
        print("  1. Using a smaller image")
        print("  2. Increasing timeout with --timeout 600")
        print("  3. Checking if the image is corrupted")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
