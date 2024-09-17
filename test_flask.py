from flask import Flask, render_template, Response
import cv2
import face_recognition
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained SVM model
with open('face_recognition_students.pkl', 'rb') as model_file:
    encodingWithKnownNames = pickle.load(model_file)

encodeListKnown, names = encodingWithKnownNames

# Initialize the webcam (or use a video file by replacing 0 with the file path)
video_capture = cv2.VideoCapture(0)

def generate_frames():
    while True:
        # Capture frame-by-frame
        success, frame = video_capture.read()

        if not success:
            break

        # Convert the frame from BGR (used by OpenCV) to RGB (used by face_recognition)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            
            matches = face_recognition.compare_faces(encodeListKnown, face_encoding)
            faceDis = face_recognition.face_distance(encodeListKnown, face_encoding)

            matchIndex = np.argmin(faceDis)

            name = "Unknown"

            if matches[matchIndex]:
                name = names[matchIndex]

            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Draw the name of the student below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame in byte format as part of an MJPEG stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    # Return the response generated along with the specific media type (mime type)
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    # Render the HTML template that displays the video stream
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
