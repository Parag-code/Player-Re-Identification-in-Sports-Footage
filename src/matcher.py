import cv2
import numpy as np

def get_histogram(image, box):
    x1, y1, x2, y2 = box
    cropped = image[y1:y2, x1:x2]
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    return cv2.normalize(hist, hist).flatten()

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def match_players(broadcast_data, tacticam_data):
    results = []
    num_frames = min(len(broadcast_data), len(tacticam_data))
    for frame_idx in range(num_frames):
        frame1, boxes1 = broadcast_data[frame_idx]
        frame2, boxes2 = tacticam_data[frame_idx]
        for box2_idx, box2 in enumerate(boxes2):
            best_score = 0
            best_match = None
            hist2 = get_histogram(frame2, box2)
            for box1_idx, box1 in enumerate(boxes1):
                hist1 = get_histogram(frame1, box1)
                score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                if score > best_score:
                    best_score = score
                    best_match = box1_idx
            results.append((frame_idx, box2_idx, best_match, best_score))
    return results
