import cv2
import face_recognition
import pickle
import numpy as np
import asyncio
import concurrent.futures

url = "http://192.168.1.22:4747/video"

# Load the trained SVM model
with open('face_recognition_model.pkl', 'rb') as model_file:
    clf = pickle.load(model_file)

# Initialize the webcam
video_capture = cv2.VideoCapture(url)

# Set frame processing variables
resize_factor = 0.5  # Scale down frame resolution to speed up processing
frame_skip = 2       # Process every 2nd frame

# Create a thread pool executor to run face recognition in a background thread
executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

def process_frame(rgb_small_frame):
    """Function to run face recognition in a separate thread."""
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
    student_names = []
    for face_encoding in face_encodings:
        face_encoding = np.array(face_encoding).reshape(1, -1)  # Reshape for the SVM model
        name = clf.predict(face_encoding)[0]
        student_names.append(name)
    
    return face_locations, student_names

async def main():
    loop = asyncio.get_event_loop()
    frame_count = 0

    while True:
        # Capture a single frame from the webcam
        ret, frame = video_capture.read()

        # Skip frames based on frame_skip value
        if not ret or (frame_count % frame_skip != 0):
            frame_count += 1
            continue

        # Reduce frame size for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)

        # Convert the frame from BGR (used by OpenCV) to RGB (used by face_recognition)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Run face recognition asynchronously in the background
        result = await loop.run_in_executor(executor, process_frame, rgb_small_frame)
        face_locations, student_names = result  # Unpack the result here after awaiting

        # Draw rectangles around detected faces
        for (top, right, bottom, left), name in zip(face_locations, student_names):
            # Scale back the face locations to match original frame size
            top = int(top / resize_factor)
            right = int(right / resize_factor)
            bottom = int(bottom / resize_factor)
            left = int(left / resize_factor)

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

        frame_count += 1

# Run the asyncio event loop
asyncio.run(main())

# Release the webcam and close OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
