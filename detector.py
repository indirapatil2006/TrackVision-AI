
import cv2
from ultralytics import YOLO


def run_detection():

    # Load the pretrained YOLO model
    print("Loading YOLO model...")
    model = YOLO("yolov8n.pt")

    # Open the default webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    print("Webcam started successfully!")
    print("Press Q to stop detection.")

    while True:

        # Read a frame from webcam
        ret, frame = cap.read()

        if not ret:
            print("Unable to read video frame.")
            break

        # Detect objects using YOLO
        results = model.predict(
            frame,
            conf=0.4,
            verbose=False
        )

        # Draw bounding boxes and labels
        annotated_frame = results[0].plot()

        # Display the processed frame
        cv2.imshow(
            "TrackVision AI - Object Detection",
            annotated_frame
        )

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    print("Object detection stopped.")


if __name__ == "__main__":
    run_detection()
