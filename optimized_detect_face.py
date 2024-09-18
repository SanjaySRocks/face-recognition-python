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

# Set the scale factor for resizing
scale_factor = 0.25  # Resize frame to 25% of original size

while True:
    # Capture a single frame from the webcam
    ret, frame = video_capture.read()
    if not ret:
        break

    # Resize the frame for faster face detection (scaling by 0.25)
    small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)

    # Convert the small frame from BGR to RGB for face_recognition
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the resized frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # Iterate over each face found in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        
        # Match the detected face with known faces
        matches = face_recognition.compare_faces(encodeListKnown, face_encoding)
        face_distances = face_recognition.face_distance(encodeListKnown, face_encoding)
        match_index = np.argmin(face_distances)

        name = "Unknown"
        if matches[match_index]:
            name = names[match_index]

        # Scale face locations back to original frame size
        top = int(top / scale_factor)
        right = int(right / scale_factor)
        bottom = int(bottom / scale_factor)
        left = int(left / scale_factor)

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw the name of the person below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    # Display the resulting frame with rectangles and names
    cv2.imshow('Face Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
