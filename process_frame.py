from pupil_apriltags import Detector
from pupil_apriltags.bindings import Detection
import numpy as np
import numpy.typing as npt
import cv2

K = np.array([
    [546.6314297825006, 0.0, 333.04330891767796],
    [0.0, 547.4164926890426, 243.95434359639907],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

dist = np.array([
    0.03729832505704915,
   -0.03571013949720708,
   -0.0006186745442605592,
    0.00023924698219720736,
   -0.04944676190628474,
   -0.0014389500705382608,
    0.0016262738185432045,
    0.002009432942358671
], dtype=np.float64)

detector = Detector(
    families="tag36h11",
    nthreads=4,
    quad_decimate=1.0,   # increase to 2.0 if you need speed
    quad_sigma=0.0,
    refine_edges=True,
    decode_sharpening=0.25
)

def _undistort(gray):
    h, w = gray.shape
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0)
    return cv2.undistort(gray, K, dist, None, new_K), new_K

def _detect_tags(gray, tag_size_meters):
    undistorted, new_K = _undistort(gray)

    fx = new_K[0, 0]
    fy = new_K[1, 1]
    cx = new_K[0, 2]
    cy = new_K[1, 2]

    detections = detector.detect(
        undistorted, # type: ignore
        estimate_tag_pose=True,
        camera_params=(fx, fy, cx, cy),
        tag_size=tag_size_meters
    )

    return detections

def process_frame(gray_img: npt.NDArray) -> dict:
    """Process a grayscale image and draw AprilTag outlines."""

    tag_size_meters = 0.1651
    tags: list[Detection] = _detect_tags(gray_img, tag_size_meters)  # type: ignore
    
    print(tags)

    # Copy the image so we don't mutate the input
    debug_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

    for tag in tags:
        corners = tag.corners.astype(int) # type: ignore

        # Draw the 4 edges
        for i in range(4):
            p1 = tuple(corners[i])
            p2 = tuple(corners[(i + 1) % 4])
            cv2.line(debug_img, p1, p2, 255, 2)

        # Draw tag ID at center
        cx, cy = map(int, tag.center) # type: ignore
        cv2.putText(
            debug_img,
            str(tag.tag_id),
            (cx - 10, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )

    return {
        "tags": tags,
        "debug_image": debug_img,
    }