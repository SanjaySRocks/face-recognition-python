import face_recognition
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pickle

# Load the trained SVM model from the pickle file
with open('face_recognition_model.pkl', 'rb') as model_file:
    clf = pickle.load(model_file)

# Define function to predict the name based on face encoding
def predict_name(face_encoding):
    # Predict the class label for the face encoding
    prediction = clf.predict([face_encoding])
    
    return prediction[0]

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
    # Predict the name using the SVM classifier
    name = predict_name(face_encoding)
        
    # Draw a rectangle around the face
    draw.rectangle(((left, top), (right, bottom)), outline="red", width=3)
    # Draw the label with the name below the rectangle
    draw.text((left, bottom + 10), name, fill="red", font=font)

# Convert Pillow image back to OpenCV format
final_image = np.array(pil_image)

# Display the image with OpenCV
cv2.imshow("Face Recognition", final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
