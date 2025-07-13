import cv2
from detector import PlayerDetector
from matcher import match_players
import os
import matplotlib.pyplot as plt
from matcher import get_histogram


FRAME_SKIP = 5
MAX_FRAMES = 100

def process_video(path, detector, frame_skip=FRAME_SKIP, max_frames=MAX_FRAMES):
    cap = cv2.VideoCapture(path)
    detections = []
    frame_idx = 0
    processed = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or processed >= max_frames:
            break
        if frame_idx % frame_skip == 0:
            boxes = detector.detect(frame)
            detections.append((frame, boxes))
            processed += 1
        frame_idx += 1
    cap.release()
    return detections

def main():
    video1_path = "data/broadcast.mp4"
    video2_path = "data/tacticam.mp4"
    model_path = "models/yolov11.pt"

    detector = PlayerDetector(model_path)

    print("Processing broadcast video...")
    broadcast_data = process_video(video1_path, detector)

    print("Processing tacticam video...")
    tacticam_data = process_video(video2_path, detector)

    print("Matching players...")
    matched = match_players(broadcast_data, tacticam_data)

    # Display HSV histograms for first 3 players in first frame of both videos
    def plot_histogram(hist, title):
        plt.figure()
        plt.title(title)
        plt.xlabel('Bin')
        plt.ylabel('Frequency')
        plt.plot(hist)
        plt.show()

    # First frame, first 3 boxes (or fewer if less detected)
    if broadcast_data and tacticam_data:
        b_frame, b_boxes = broadcast_data[0]
        t_frame, t_boxes = tacticam_data[0]
        print("\nDisplaying histograms for first 3 players in first frame of both videos...")
        for idx, box in enumerate(b_boxes[:3]):
            hist = get_histogram(b_frame, box)
            plot_histogram(hist, f"Broadcast Frame 0, Player {idx} HSV Histogram")
        for idx, box in enumerate(t_boxes[:3]):
            hist = get_histogram(t_frame, box)
            plot_histogram(hist, f"Tacticam Frame 0, Player {idx} HSV Histogram")

    # Print results in a readable format
    print("\nMatched Players (Tacticam → Broadcast):")
    last_frame = -1
    for frame_index, tacticam_box_index, matched_broadcast_box_index, score in matched:
        if matched_broadcast_box_index is None:
            if last_frame != frame_index:
                print(f"Frame: {frame_index} -- No broadcast detections found!")
                last_frame = frame_index
            print(f"  Tacticam Box: {tacticam_box_index} → No match (Score: {score:.3f})")
        else:
            print(f"Frame: {frame_index}, Tacticam Box: {tacticam_box_index} → Broadcast Box: {matched_broadcast_box_index}, Score: {score:.3f}")

    print("\nDone. Matched players printed above.")

if __name__ == "__main__":
    main()
