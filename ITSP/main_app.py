from flask import Flask, request, jsonify, render_template, Response
import sqlite3
import cv2
import numpy as np
import dlib
import os
import base64
import time
import threading
import queue
import logging
from logging.handlers import RotatingFileHandler

app = Flask(__name__, template_folder='templates')

def init_db():
    conn = sqlite3.connect('location.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS location
                      (id INTEGER PRIMARY KEY, latitude REAL, longitude REAL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
log_file = 'app.log'
file_handler = RotatingFileHandler(log_file, maxBytes=1024*1024, backupCount=10)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)
app.logger.setLevel(logging.INFO)
app.logger.addHandler(file_handler)

facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

face_descriptors = {}
labels = []

unrecognized_face = None
video_capture = None
frame_queue = queue.Queue(maxsize=1)
event_queue = queue.Queue()
capture_running = False
input_requested = False  
last_unrecognized_time = 0
last_input_prompt_time = 0

images_dir = "Face Photo"

def load_face_descriptors(images_dir):
    for person_dir in os.listdir(images_dir):
        person_path = os.path.join(images_dir, person_dir)
        if os.path.isdir(person_path):
            face_descriptors[person_dir] = []
            for filename in os.listdir(person_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    image_path = os.path.join(person_path, filename)
                    img = cv2.imread(image_path)
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    faces = detector(rgb_img)
                    if len(faces) == 1:
                        face = faces[0]
                        shape = predictor(rgb_img, face)
                        face_descriptor = facerec.compute_face_descriptor(rgb_img, shape)
                        face_descriptor_np = np.array(face_descriptor)
                        face_descriptors[person_dir].append(face_descriptor_np)
                        labels.append(person_dir)
                    else:
                        app.logger.warning(f"Skipping {image_path}: No face detected or multiple faces detected.")

load_face_descriptors(images_dir)

def compare_faces(known_face_descriptors, face_descriptor, tolerance=0.5):
    recognized_labels = []
    for label, descriptors in known_face_descriptors.items():
        if descriptors:  
            distances = np.linalg.norm(descriptors - face_descriptor, axis=1)
            min_distance_idx = np.argmin(distances)
            if distances[min_distance_idx] <= tolerance:
                recognized_labels.append(label)
    if recognized_labels:
        return recognized_labels[0]
    else:
        return "Not Recognized"

def capture_video():
    global capture_running, unrecognized_face, last_unrecognized_time
    cap = cv2.VideoCapture(0)
    while capture_running:
        success, frame = cap.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            for face in faces:
                shape = predictor(frame, face)
                face_descriptor = facerec.compute_face_descriptor(frame, shape)
                face_descriptor = np.array(face_descriptor)
                label = compare_faces(face_descriptors, face_descriptor)
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                color = (0, 255, 0) if label != "Not Recognized" else (0, 0, 255)
                x_new = max(0, x - w // 2)
                y_new = max(0, y - h // 2)

                w *= 2
                h *= 2

                cv2.rectangle(frame, (x_new, y_new), (x_new + w, y_new + h), color, 2)

                cv2.putText(frame, label, (x_new, y_new - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                if label == "Not Recognized":
                    current_time = time.time()
                    if current_time - last_unrecognized_time >= 5:
                        if frame_queue.qsize() < 1:  
                            unrecognized_face = frame[y_new:y_new+h, x_new:x_new+w].copy()  
                            last_unrecognized_time = current_time
                            event_queue.put("unrecognized")  
                        else:
                            app.logger.warning("Reached maximum queued photos limit. Ignoring new unrecognized face.")

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            if not frame_queue.full():
                frame_queue.put(frame)

    cap.release()

@app.route('/')
def index():
    global input_requested
    input_requested = True  
    app.logger.info('Accessed index page')
    return render_template("index4.html")

@app.route('/map')
def map_index():
    return render_template('map_index.html')

@app.route('/share_location')
def share_location():
    return render_template('share_location.html')

@app.route('/update_location', methods=['POST'])
def update_location():
    data = request.get_json()
    latitude = data.get('latitude')
    longitude = data.get('longitude')
    
    conn = sqlite3.connect('location.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO location (latitude, longitude) VALUES (?, ?)", (latitude, longitude))
    conn.commit()
    conn.close()
    return jsonify({"status": "success"})

@app.route('/get_location', methods=['GET'])
def get_location():
    conn = sqlite3.connect('location.db')
    cursor = conn.cursor()
    cursor.execute("SELECT latitude, longitude, timestamp FROM location ORDER BY id DESC LIMIT 1")
    location = cursor.fetchone()
    conn.close()
    if location:
        return jsonify({"latitude": location[0], "longitude": location[1], "timestamp": location[2]})
    else:
        return jsonify({"error": "No location data available"}), 404

@app.route('/video_feed')
def video_feed():
    global capture_running, video_capture
    if not capture_running:
        video_capture = cv2.VideoCapture(0)
        capture_running = True
        threading.Thread(target=capture_video, daemon=True).start()

    def generate():
        while True:
            if not frame_queue.empty():
                frame = frame_queue.get()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                time.sleep(0.1)  

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/sse')
def sse():
    def event_stream():
        global last_input_prompt_time
        while True:
            try:
                event = event_queue.get(timeout=1)
                if event == "unrecognized":
                    current_time = time.time()
                    if current_time - last_input_prompt_time >= 10:
                        ret, buffer = cv2.imencode('.jpg', unrecognized_face)
                        if ret:
                            unrecognized_face_bytes = base64.b64encode(buffer).decode('utf-8')
                            yield f"event: {event}\ndata: {unrecognized_face_bytes}\n\n"
                            last_input_prompt_time = current_time
                else:
                    yield f"event: {event}\n\n"
            except queue.Empty:
                yield ": heartbeat\n\n"
    return Response(event_stream(), mimetype='text/event-stream')

@app.route('/input_name', methods=['POST'])
def input_name():
    global unrecognized_face, input_requested, frame_queue
    name = request.form['name']
    if input_requested and name:
        if name not in face_descriptors:
            face_descriptors[name] = []

        if unrecognized_face is not None:
            rgb_face = cv2.cvtColor(unrecognized_face, cv2.COLOR_BGR2RGB)
            faces = detector(rgb_face)
            if len(faces) == 1:
                face = faces[0]
                shape = predictor(rgb_face, face)
                face_descriptor = facerec.compute_face_descriptor(rgb_face, shape)
                face_descriptor_np = np.array(face_descriptor)
                face_descriptors[name].append(face_descriptor_np)
                labels.append(name)

                save_dir = os.path.join(images_dir, name)
                os.makedirs(save_dir, exist_ok=True)
                existing_photos = os.listdir(save_dir)
                photo_no = len(existing_photos) + 1
                save_path = os.path.join(save_dir, f"{name}_{photo_no}.jpg")
                cv2.imwrite(save_path, unrecognized_face)

                app.logger.info(f"New face added: {name}")
                
                frame_queue = queue.Queue(maxsize=1)
                
                return jsonify({"status": "success"})
            else:
                app.logger.error("No face detected in the saved image.")
                return jsonify({"status": "error", "message": "No face detected in the saved image."})
        else:
            app.logger.error("No unrecognized face was captured.")
            return jsonify({"status": "error", "message": "No unrecognized face was captured."})
    else:
        app.logger.error("Invalid request")
        return jsonify({"status": "error", "message": "Invalid request"})


@app.route('/ignore_face', methods=['POST'])
def ignore_face():
    global unrecognized_face
    unrecognized_face = None
    return jsonify({"status": "ignored"})

init_db()

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
