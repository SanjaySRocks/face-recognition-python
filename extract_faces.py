import cv2
import face_recognition
import os

def extract_faces(input_image_path, output_folder, new_size=(128, 128)):
    # Load the image file into a numpy array
    image = face_recognition.load_image_file(input_image_path)
    
    # Find all face locations in the image
    face_locations = face_recognition.face_locations(image)
    
    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop over each face found in the image
    for i, (top, right, bottom, left) in enumerate(face_locations):
        # Extract the face image
        face_image = image[top:bottom, left:right]

        # Convert the face image from RGB (face_recognition uses RGB) to BGR (OpenCV uses BGR)
        face_image_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)

        # Resize the face image
        face_image_resized = cv2.resize(face_image_bgr, new_size)

        # Save the face image to the output folder
        output_path = os.path.join(output_folder, f"face_{i + 1}.jpg")
        cv2.imwrite(output_path, face_image_resized)
        print(f"Saved resized face {i + 1} to {output_path}")

# Usage
input_image_path = 'input_image.jpg'  # Replace with the path to your input image file
output_folder = 'output_faces'  # Replace with the path to your desired output folder
new_size = (256, 256)  # Desired size for the resized face images

extract_faces(input_image_path, output_folder, new_size)
