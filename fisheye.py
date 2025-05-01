import cv2
from pupil_apriltags import Detector
import numpy as np

# ELP USB camera calibration parameters (replace with your calibration values)
K = np.array(
    [[800.0, 0.0, 512.0], [0.0, 800.0, 384.0], [0.0, 0.0, 1.0]]
)  # Intrinsic matrix (fx, fy, cx, cy filled with example values)
D = np.array(
    [0.1, -0.05, 0.0, 0.0]
)  # Distortion coefficients (replace with actual values)
DIM = (1024, 768)  # Image dimensions

# Initialize the AprilTag detector
at_detector = Detector(
    families="tag16h5",
    quad_sigma=0.0,
    quad_decimate=1.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=1,
)

cap = cv2.VideoCapture(
    "output2.mp4"
)  # Use the ELP USB camera (usually device 0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from camera.")
        break

    # Undistort the frame for fisheye lens
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K, DIM, cv2.CV_16SC2
    )
    undistorted_frame = cv2.remap(
        frame,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    # Convert the frame to grayscale
    gray = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)

    # Detect AprilTags in the frame
    tags = at_detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=(K[0, 0], K[1, 1], K[0, 2], K[1, 2]),
        tag_size=0.15,  # Tag size in meters (adjust as needed)
    )

    if tags != []:
        for tag in tags:
            if (tag.tag_id != 0) and (tag.tag_id != 1) and (tag.tag_id != 3):
                continue
            if tag.decision_margin > 42:
                # Draw a rectangle around the detected tag
                cv2.polylines(
                    undistorted_frame,
                    [tag.corners.astype(int)],
                    isClosed=True,
                    color=(0, 255, 0),
                    thickness=2,
                )
                # Print the tag ID in the bottom-right corner
                cv2.putText(
                    undistorted_frame,
                    f"Tag ID: {tag.tag_id}",
                    (
                        undistorted_frame.shape[1] - 150,
                        undistorted_frame.shape[0] - 20,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )
                # Print the decision margin in the bottom-left corner
                cv2.putText(
                    undistorted_frame,
                    f"Margin: {tag.decision_margin:.2f}",
                    (10, undistorted_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                )
                # Display the frame with detected tags
                cv2.imshow("Detected Tags", undistorted_frame)

    # Display the undistorted frame in a window
    cv2.imshow("Live Feed", gray)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
