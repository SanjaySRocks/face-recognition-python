import os
import face_recognition
import pickle

# Path to the dataset folder containing student images
dataset_folder = "dataset.students"
model_filename = "face_recoginition_students.pkl"

# Create empty lists to store encodings and names
encodings = []
names = []

# Loop through each folder in the dataset
for student_name in os.listdir(dataset_folder):
    student_folder = os.path.join(dataset_folder, student_name)
    
    # Loop through each image in the student's folder
    for img_filename in os.listdir(student_folder):
        img_path = os.path.join(student_folder, img_filename)
        
        print("Processing: ",img_path)
        
        # Load the image and encode the face
        img = face_recognition.load_image_file(img_path)
        img_encodings = face_recognition.face_encodings(img)
        
        if len(img_encodings) > 0:
            encodings.append(img_encodings[0])  # Only take the first encoding (assuming one face per image)
            names.append(student_name)

encodingWithKnownNames = [encodings, names]

# Save the trained model
with open(model_filename, 'wb') as model_file:
    pickle.dump(encodingWithKnownNames, model_file)
