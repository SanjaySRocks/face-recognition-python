import cv2
import face_recognition

img = cv2.imread("students_images/Elon Musk.jpg")
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_encoding = face_recognition.face_encodings(rgb_img)[0]

img2 = cv2.imread("students_images/Elon Musk.jpg")
rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img_encoding2 = face_recognition.face_encodings(rgb_img2)[0]

result = face_recognition.compare_faces([img_encoding], img_encoding2)
print("Result: ", result)



# import face_recognition
# known_image = face_recognition.load_image_file("sanjay singh.jpg")
# unknown_image = face_recognition.load_image_file("sanjay singh.jpg")

# biden_encoding = face_recognition.face_encodings(known_image)[0]
# unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

# results = face_recognition.compare_faces([biden_encoding], unknown_encoding)