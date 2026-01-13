from video_sources.video_source import VideoSource
import numpy.typing as npt
import cv2

class LaptopVideoSource(VideoSource):
    """Video source with usb cam plugged into laptop."""
    
    def __init__(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) # DirectShow mode
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -11)     # try moving this slider

    def get_frame(self) -> npt.NDArray | None:
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame