import cv2

# Replace the IP address below with the one shown in your IP Webcam app
url = "http://192.168.1.22:4747/video"  # Use your phone's IP here

# Initialize the video capture with the IP Webcam stream
video_capture = cv2.VideoCapture(url)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    if ret:
        # Display the resulting frame
        cv2.imshow('Droid Camera', frame)

    # Break from the loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
