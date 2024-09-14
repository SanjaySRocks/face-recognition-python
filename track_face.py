import cv2

# Load the pre-trained face detection model
face_cap = cv2.CascadeClassifier(r"C:\Users\sanjay\Documents\GitHub\face-recognition-python\env\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")

# Start capturing video from the webcam
video_cap = cv2.VideoCapture(0)

while True:
    ret, video_data = video_cap.read()

    # Convert video data to black and white
    gray_frame = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)

    faces = face_cap.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x,y,w,h) in faces:
        cv2.rectangle(video_data, (x,y), (x+w, y+h), (255,255,255), 2)

    cv2.imshow("Face Capture", video_data)

    if cv2.waitKey(10) == 27: # esc key
        break


video_cap.release()
cv2.destroyAllWindows()