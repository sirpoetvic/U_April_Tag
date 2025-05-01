import cv2
from pupil_apriltags import Detector

# Initialize the AprilTag detector
at_detector = Detector(
    families="tag16h5",
    quad_sigma=0.0,
    quad_decimate=1.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=1,
)

# Load the image
image_path = "image copy 2.png"
image = cv2.imread(image_path)
if image is None:
    print("Error: Could not load image.")
    exit()

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Resize the image to 640x480
gray = cv2.resize(gray, (640, 480))
# Turn gray into uint8
gray = gray.astype("uint8")

# Detect AprilTags in the image
tags = at_detector.detect(
    gray,
    # estimate_tag_pose=True,
    # camera_params=(1.0, 1.0, 1.0, 1.0),
    # tag_size=0.15,  # Tag size in meters (adjust as needed)
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
                image,
                [tag.corners.astype(int)],
                isClosed=True,
                color=(0, 255, 0),
                thickness=2,
            )
            # Print the tag ID next to the detected tag
            tag_center = tag.center.astype(int)
            cv2.putText(
                image,
                f"ID: {tag.tag_id}",
                (tag_center[0] + 10, tag_center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )
            # Print the decision margin next to the detected tag
            cv2.putText(
                image,
                f"Margin: {tag.decision_margin:.2f}",
                (tag_center[0] + 10, tag_center[1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )

# Display the image with detected tags
cv2.imshow("Detected Tags", image)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
