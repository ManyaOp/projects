import cv2
import dlib
import numpy as np
import os

# Load pre-trained face recognition model
facerec = dlib.face_recognition_model_v1(r"C:\Users\manya\OneDrive - Indian Institute of Technology Bombay\ITSP\dlib_face_recognition_resnet_model_v1.dat")

# Load face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"shape_predictor_68_face_landmarks.dat")

# Directory containing images of faces
images_dir = r"C:\Users\manya\OneDrive - Indian Institute of Technology Bombay\ITSP\Face Photo"

# Initialize dictionaries to store face descriptors for each person
face_descriptors = {}
labels = []

# Loop through each person's directory in the image directory
for person_dir in os.listdir(images_dir):
    person_path = os.path.join(images_dir, person_dir)
    if os.path.isdir(person_path):
        # Initialize list to store face descriptors for this person
        face_descriptors[person_dir] = []
        
        # Loop through each image in the person's directory
        for filename in os.listdir(person_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(person_path, filename)
                img = cv2.imread(image_path)
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Detect faces in the image
                faces = detector(rgb_img)

                # Assume there's only one face in each image for simplicity
                if len(faces) == 1:
                    face = faces[0]
                    # Get the face descriptor
                    shape = predictor(rgb_img, face)
                    face_descriptor = facerec.compute_face_descriptor(rgb_img, shape)
                    # Convert to numpy array for easier handling
                    face_descriptor_np = np.array(face_descriptor)
                    face_descriptors[person_dir].append(face_descriptor_np)
                    # Add label for this person
                    labels.append(person_dir)
                else:
                    print(f"Skipping {image_path}: No face detected or multiple faces detected.")

# Convert lists to numpy arrays
labels = np.array(labels)

# Save face descriptors and labels to a file (optional)
np.savez("faces_database.npz", face_descriptors=face_descriptors, labels=labels)

# Function to compare face descriptors
def compare_faces(known_face_descriptors, face_descriptor, tolerance=0.5):
    recognized_labels = []
    for label, descriptors in known_face_descriptors.items():
        distances = np.linalg.norm(descriptors - face_descriptor, axis=1)
        min_distance_idx = np.argmin(distances)
        if distances[min_distance_idx] <= tolerance:
            recognized_labels.append(label)
    if recognized_labels:
        return recognized_labels[0]  # Return the label of the closest match
    else:
        return "Not Recognized"  # If no match found within tolerance, return "Not Recognized"


# Capture video from the default camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale image
    faces = detector(gray)
    
    # Loop over each face
    for face in faces:
        # Get the face descriptor
        shape = predictor(frame, face)
        face_descriptor = facerec.compute_face_descriptor(frame, shape)
        face_descriptor = np.array(face_descriptor)
        
        # Compare with known faces and get the label
        label = compare_faces(face_descriptors, face_descriptor)
        
        # Draw a rectangle around the face
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        color = (0, 255, 0) if label != "Not Recognized" else (0, 0, 255)  # Green if recognized, red if not recognized
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Display the result
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # If face is not recognized, prompt user for name and save the photo
        if label == "Not Recognized":
            name = input("Enter the name of the unrecognized person: ")
            if name:
                if name not in face_descriptors:
                    face_descriptors[name] = [face_descriptor]
                else:
                    face_descriptors[name].append(face_descriptor)
                    
                save_dir = os.path.join(images_dir, name)
                os.makedirs(save_dir, exist_ok=True)
                existing_photos = os.listdir(save_dir)
                if existing_photos:
                    photo_no = max([int(photo.split("_")[1].split(".")[0]) for photo in existing_photos]) + 1
                else:
                    photo_no = 1
                save_path = os.path.join(save_dir, f"{name}_{photo_no}.jpg")
                cv2.imwrite(save_path, frame[y:y+h, x:x+w])
                print(f"Saved photo of {name} in {save_path}")
                
    # Display the frame
    cv2.imshow('Face Recognition', frame)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Save unrecognized face descriptors to files
for name, descriptors in face_descriptors.items():
    np.savez(os.path.join(images_dir, name + "_faces.npz"), face_descriptors=np.array(descriptors))
