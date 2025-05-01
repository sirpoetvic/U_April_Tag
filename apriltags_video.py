import cv2
from pupil_apriltags import Detector

# Initialize the AprilTag detector
# family is tag16h5
# quad_decimate: controls resolution. higher resolution = more accurate, but slower
# quad_sigma: lower values allow for more noise (0.0 disables blurring),
# something like 0.8 is good for noisy images

at_detector = Detector(
    families="tag16h5",
    quad_sigma=0.0,
    quad_decimate=1.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=1,
)

cap = cv2.VideoCapture("output2.mp4")
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()
while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from camera.")
        break
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Resize the frame to 640x480
    gray = cv2.resize(gray, (640, 480))
    # Turn gray into uint8
    gray = gray.astype("uint8")
    # Detect AprilTags in the frame
    tags = at_detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=(1.0, 1.0, 1.0, 1.0),
        tag_size=0.15,  # Tag size in meters (adjust as needed)
    )
    if tags != []:
        for tag in tags:
            if (tag.tag_id != 0) and (tag.tag_id != 1) and (tag.tag_id != 3):
                continue
            print(tag)
            # Low margin is okay because we only use valid tags, and none are similar
            if tag.decision_margin > 42:
                # Draw a rectangle around the detected tag
                cv2.polylines(
                    frame,
                    [tag.corners.astype(int)],
                    isClosed=True,
                    color=(0, 255, 0),
                    thickness=2,
                )
                # Print the tag ID in the bottom-right corner
                cv2.putText(
                    frame,
                    f"Tag ID: {tag.tag_id}",
                    (frame.shape[1] - 150, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )
                # Print the decision margin in the bottom-left corner
                cv2.putText(
                    frame,
                    f"Margin: {tag.decision_margin:.2f}",
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                )
                # Display the frame with detected tags
                cv2.imshow("Detected Tags", frame)

    # Display the frame in a window
    cv2.imshow("Live Feed", gray)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
