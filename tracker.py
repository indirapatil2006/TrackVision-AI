
from deep_sort_realtime.deepsort_tracker import DeepSort


class ObjectTracker:

    def __init__(self):

        # Initialize Deep SORT
        self.tracker = DeepSort(
            max_age=30,
            n_init=2,
            max_cosine_distance=0.3
        )

    def update(self, detections, frame):

        # Update tracker with YOLO detections
        tracks = self.tracker.update_tracks(
            detections,
            frame=frame
        )

        return tracks
