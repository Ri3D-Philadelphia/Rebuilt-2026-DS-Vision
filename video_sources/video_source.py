from abc import ABC, abstractmethod
import numpy.typing as npt

class VideoSource(ABC):
    """Interface class for a video source."""

    @abstractmethod
    def get_frame(self) -> npt.NDArray | None:
        """Returns the next frame from the video source if it exists, otherwise `None`."""
        pass
