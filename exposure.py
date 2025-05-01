import cv2
from matplotlib import pyplot as plt

# Load video
video_path = "output.mp4"
output_path = "filtered_output.mp4"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for output video

# Create VideoWriter object
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply filters
    darker = cv2.convertScaleAbs(frame, alpha=0.8, beta=-40)
    denoised = cv2.fastNlMeansDenoisingColored(darker, None, 10, 10, 7, 21)

    # Write the processed frame to the output video
    out.write(denoised)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Filtered video saved to {output_path}")
