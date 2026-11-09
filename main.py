
import cv2
import os
from ultralytics import YOLO
from tracker import ObjectTracker

# Choose input source
print("\n===== TrackVision AI =====")
print("1. Webcam")
print("2. Video file")

choice = input("Enter your choice (1 or 2): ")

if choice == "1":
    source = 0
    output_path = "output/tracked_webcam.mp4"
elif choice == "2":
    source = "videos/test_video.mp4"
    output_path = "output/tracked_test_video.mp4"
else:
    print("Invalid choice!")
    exit()

# Create output folder if it doesn't exist
os.makedirs("output", exist_ok=True)

# Load YOLO model
print("\nLoading YOLO model...")
model = YOLO("yolov8n.pt")

# Initialize tracker
tracker = ObjectTracker()

# Open webcam or video
cap = cv2.VideoCapture(source)

if not cap.isOpened():
    print("Error: Could not open the selected video source.")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 20.0

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create output video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)

print("\nTrackVision AI started!")
print("Press Q to stop.")

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Video finished or frame unavailable.")
            break

        # Detect objects using YOLO
        results = model.predict(
            frame,
            conf=0.4,
            imgsz=320,
            device="cpu",
            verbose=False
        )

        detections = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                bbox = [
                    x1,
                    y1,
                    x2 - x1,
                    y2 - y1
                ]

                detections.append(
                    (bbox, confidence, class_name)
                )

        # Update Deep SORT tracker
        tracks = tracker.update(detections, frame)

        # Draw tracking boxes and IDs
        for track in tracks:
            if not track.is_confirmed():
                continue

            x1, y1, x2, y2 = map(
                int, track.to_ltrb()
            )

            track_id = track.track_id
            class_name = track.get_det_class()

            if class_name is None:
                class_name = "Object"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            label = f"{class_name} ID: {track_id}"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # Save processed frame
        video_writer.write(frame)

        # Display output
        cv2.imshow("TrackVision AI - Object Tracking", frame)

        # Press Q to stop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Tracking stopped.")
            break

finally:
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

print("\nProcessed video saved at:", output_path)
