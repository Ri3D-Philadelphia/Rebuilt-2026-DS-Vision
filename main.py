from video_sources.video_source import VideoSource
from video_sources.laptop_video_source import LaptopVideoSource
from video_sources.rio_video_source import RioVideoSource
from pupil_apriltags.bindings import Detection

from process_frame import process_frame, rotation_matrix_to_euler_angles

from networktables import NetworkTables

import time
import cv2

# video_source: VideoSource = LaptopVideoSource()
video_source: VideoSource = RioVideoSource()


NetworkTables.initialize(server="10.17.40.2")  # 10.TE.AM.2
# NetworkTables.initialize(server="roboRIO-1740-FRC.local")

table = NetworkTables.getTable("vision")

# Wait until connected
print("Trying to connect to networktables...")
while not NetworkTables.isConnected():
    time.sleep(0.1)
print("Connected")

while True:
    frame = video_source.get_frame()
    if frame is None:
        continue
    
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    data = process_frame(gray_img)
    
    tags: list[Detection] = data["tags"]

    if len(tags) == 0:
        table.putBoolean("Tag Detected", False)
    else:
        table.putBoolean("Tag Detected", True)
        for i, tag in enumerate(tags):
            cx, cy = tag.center # type: ignore
            roll, pitch, yaw = rotation_matrix_to_euler_angles(tag.pose_R)
            table.putNumberArray(f"tag{i}", [
                cx, 
                cy, 
                tag.pose_t[2][0], # type: ignore
                -tag.pose_t[0][0], # type: ignore
                yaw
            ])
            table.putString(f"tag{i}pose_r", str(tag.pose_R))
            table.putString(f"tag{i}pose_t", str(tag.pose_t))

    # Show windows
    cv2.imshow("Camera", frame)
    cv2.imshow("Annotated", data["debug_image"])

    # REQUIRED for window updates
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cv2.destroyAllWindows()