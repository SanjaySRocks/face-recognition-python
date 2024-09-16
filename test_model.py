import face_recognition
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pickle

# Load the trained model
with open('face_recognition_students.pkl', 'rb') as model_file:
    encodingWithKnownNames = pickle.load(model_file)

encodeListKnown, names = encodingWithKnownNames


# Load the input image
input_image_path = "input_image.jpg"
input_image = face_recognition.load_image_file(input_image_path)

# Find all face locations and encodings in the input image
face_locations = face_recognition.face_locations(input_image)
face_encodings = face_recognition.face_encodings(input_image, face_locations)

# Convert image to BGR format for OpenCV
input_image_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)

# Create a Pillow image for drawing
pil_image = Image.fromarray(input_image_bgr)
draw = ImageDraw.Draw(pil_image)

# Load a font for drawing text
try:
    font = ImageFont.truetype("arial.ttf", 36)
except IOError:
    font = ImageFont.load_default()

for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    
    matches = face_recognition.compare_faces(encodeListKnown, face_encoding)
    faceDis = face_recognition.face_distance(encodeListKnown, face_encoding)

    matchIndex = np.argmin(faceDis)

    # print(matches)
    # print(faceDis)


    if matches[matchIndex]:
        name = names[matchIndex]
    else:
        name = "Unknown"

    # Draw a rectangle around the face
    draw.rectangle(((left, top), (right, bottom)), outline="red", width=3)

    # Calculate text size
    text_bbox = draw.textbbox((left, bottom + 10), name, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Draw background rectangle for text with blue color
    draw.rectangle(((left, bottom + 10), (left + text_width, bottom + 10 + text_height)), fill="red")

    # Draw the label with the name below the rectangle with white text
    draw.text((left, bottom + 10), name, fill="white", font=font)

# Convert Pillow image back to OpenCV format
final_image = np.array(pil_image)

# Resize image to fit within the window while preserving the aspect ratio
max_width = 1024  # Maximum width of the display window
max_height = 1024  # Maximum height of the display window
height, width, _ = final_image.shape
aspect_ratio = width / height

if width > max_width or height > max_height:
    if width > height:
        new_width = max_width
        new_height = int(max_width / aspect_ratio)
    else:
        new_height = max_height
        new_width = int(max_height * aspect_ratio)

    final_image_resized = cv2.resize(final_image, (new_width, new_height))
else:
    final_image_resized = final_image

# Display the image with OpenCV
cv2.imshow("Face Recognition", final_image_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()