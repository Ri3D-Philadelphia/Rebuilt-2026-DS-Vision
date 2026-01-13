from video_sources.video_source import VideoSource
import numpy.typing as npt
import cv2

class RioVideoSource(VideoSource):
    """Video source from Rio."""
    
    def __init__(self):
        print("Trying to connect to RIO video stream")
        self.cap = cv2.VideoCapture("http://roborio-1740-frc.local:1181/stream.mjpg")
        print("Connected to RIO video stream")

    def get_frame(self) -> npt.NDArray | None:
        # ret, frame = self.cap.read()
        # if not ret:
        #     return None
        # return frame
    
        # Drain buffered frames
        for _ in range(5):  # 2–5 is usually enough
            if not self.cap.grab():
                break

        ret, frame = self.cap.retrieve()
        if not ret:
            return None
        return frame