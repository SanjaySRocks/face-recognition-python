import face_recognition
import cv2
import os

# Path to your folder containing student images
students_images_folder = "students_images"

# Function to load images from the folder and encode faces
def load_student_encodings(folder_path):
    student_encodings = []
    student_names = []

    for filename in os.listdir(folder_path):
        print(filename)
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Load each image file and extract the student's name from the filename
            img_path = os.path.join(folder_path, filename)

            img = face_recognition.load_image_file(img_path)
            img_encoding = face_recognition.face_encodings(img)[0]

            # Append the encoding and student name (filename without extension)
            student_encodings.append(img_encoding)
            student_names.append(os.path.splitext(filename)[0])

    return student_encodings, student_names

# Load all students' encodings
known_face_encodings, known_face_names = load_student_encodings(students_images_folder)

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture a single frame from the webcam
    ret, frame = video_capture.read()

    # Convert the frame from BGR (used by OpenCV) to RGB (used by face_recognition)
    rgb_frame = frame[:, :, ::-1]

    # Detect faces and their encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the detected face with known student encodings
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # If a match is found, use the name of the first match
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = face_distances.argmin()
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw the name of the person below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Break from the loop with 'q' key
    key = cv2.waitKey(1)
    if key == 27:
        break
    
# Release the webcam and close OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
