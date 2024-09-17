import cv2
import face_recognition
import pickle
import numpy as np
import time 

# Load the trained SVM model
with open('face_recognition_students.pkl', 'rb') as model_file:
    encodingWithKnownNames = pickle.load(model_file)

encodeListKnown, names = encodingWithKnownNames

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

scale_factor = 0.25

fps = 0
frame_counter = 0
start_time = time.time()

while True:
    # Capture a single frame from the webcam
    success, frame = video_capture.read()

    if not success:
        break
    
    small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)

    # Convert the frame from BGR (used by OpenCV) to RGB (used by face_recognition)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

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

        top = int(top / scale_factor)
        right = int(right / scale_factor)
        bottom = int(bottom / scale_factor)
        left = int(left / scale_factor)

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw the name of the student below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

        # Calculate FPS every 1 second
    frame_counter += 1
    if (time.time() - start_time) > 1:
        fps = frame_counter / (time.time() - start_time)
        frame_counter = 0
        start_time = time.time()

    # Put the FPS text on the frame
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Break from the loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
