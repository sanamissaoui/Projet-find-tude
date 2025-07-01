def login_face():
    try:
        data = request.get_json()
        image_data = data.get('image_data')

        if not image_data:
            return jsonify({"success": False, "message": "Image manquante"}), 400

        encoded = image_data.split(",")[1]
        img_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if not detect_blink(img):
            return jsonify({"success": False, "message": "Liveness non vérifié : aucun clignement détecté."})

        unknown_faces = face_recognition.face_encodings(rgb_img)
        if not unknown_faces:
            return jsonify({"success": False, "message": "Aucun visage détecté."}), 200

        unknown_encoding = unknown_faces[0]

        images_dir = "images"
        for person_name in os.listdir(images_dir):
            person_dir = os.path.join(images_dir, person_name)
            for img_file in os.listdir(person_dir):
                known_img_path = os.path.join(person_dir, img_file)
                known_img = face_recognition.load_image_file(known_img_path)
                known_faces = face_recognition.face_encodings(known_img)

                if known_faces and face_recognition.compare_faces([known_faces[0]], unknown_encoding)[0]:
                    return jsonify({"success": True, "message": f"{person_name}"})

        return jsonify({"success": False, "message": "Aucun visage correspondant trouvé."})

    except Exception as e:
        return jsonify({"success": False, "message": "Erreur interne : " + str(e)}), 500

# --- Face blink detection setup ---
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # required

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def detect_blink(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for rect in rects:
        shape = predictor(gray, rect)
        coords = np.zeros((68, 2), dtype="int")
        for i in range(68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        leftEye = coords[42:48]
        rightEye = coords[36:42]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        if ear < 0.2:
            return True
    return False