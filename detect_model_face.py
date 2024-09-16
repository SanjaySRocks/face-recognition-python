import cv2
import face_recognition
import pickle
import numpy as np

# Load the trained SVM model
with open('face_recognition_students.pkl', 'rb') as model_file:
    encodingWithKnownNames = pickle.load(model_file)

encodeListKnown, names = encodingWithKnownNames

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture a single frame from the webcam
    ret, frame = video_capture.read()

    # small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    # Convert the frame from BGR (used by OpenCV) to RGB (used by face_recognition)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        
        matches = face_recognition.compare_faces(encodeListKnown, face_encoding)
        faceDis = face_recognition.face_distance(encodeListKnown, face_encoding)

        matchIndex = np.argmin(faceDis)

        print(matches)
        print(faceDis)

        name = "Unknown"

        if matches[matchIndex]:
            name = names[matchIndex]

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw the name of the student below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Break from the loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
