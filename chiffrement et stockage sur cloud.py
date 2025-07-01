from flask import Flask, render_template, request, jsonify, redirect
import os
import cv2
import base64
import numpy as np
import face_recognition
import threading
import webbrowser
from scipy.spatial import distance as dist
import dlib
from sqlalchemy import create_engine
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import boto3
from Crypto.Cipher import AES
import base64
import csv
import uuid
from datetime import datetime
from io import StringIO
from Crypto.Util.Padding import pad
from threading import Thread
from Crypto.Util.Padding import unpad

app = Flask(__name__, template_folder='templates', static_folder='static')

# --- AWS S3 Configuration ---

BUCKETS = ['fragment1pfe', 'fragment2pfe', 'fragment3pfe', 'redpfe']

s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)
S3_BUCKET_NAME = 'redpfe'

@app.route('/register', methods=['POST'])
def register():
    # Get form data
    nom = request.form.get('nom')
    prenom = request.form.get('prenom')
    email = request.form.get('email')
    password = request.form.get('password')
    confirm_password = request.form.get('confirmPassword')
    telephone = request.form.get('telephone')

    # Basic password confirmation check
    if password != confirm_password:
        return "Erreur: Les mots de passe ne correspondent pas.", 400

    # Generate a unique user ID
    user_id = str(uuid.uuid4())
    thread = Thread(target=fragment_data, args=(user_id,))
    thread.start()
    # Create a unique filename for S3
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    filename = f"register_{user_id}_{timestamp}.csv"

    # Create CSV content in memory
    csv_buffer = StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(['ID', 'Nom', 'Prénom', 'Email', 'Téléphone', 'Password'])
    writer.writerow([user_id, nom, prenom, email, telephone, password])
    csv_content = csv_buffer.getvalue()

    try:
        # Upload directly to S3 without saving to disk
        s3.put_object(Bucket=S3_BUCKET_NAME, Key=filename, Body=csv_content)
    except Exception as e:
        return f"Erreur lors du téléchargement sur S3: {str(e)}", 500

    return redirect('/inscri')
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
def fragment_data(user_id):
    import boto3
    import csv
    import base64
    from Crypto.Cipher import AES
    from Crypto.Util.Padding import pad

    # --- AWS S3 and AES Config for encryptor ---
    

    s3 = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION
    )

    # Nom des buckets
    buckets = {
        'frag1': 'fragment1pfe',
        'frag2': 'fragment2pfe',
        'frag3': 'fragment3pfe',
        'redpfe': 'redpfe'  # bucket redondance
    }

    # Fichiers locaux
    input_file = 'classified_output.csv'
    full_encrypted_file = 'full_encrypted_pfe.csv'  # fichier complet chiffré
    fragment1_file = 'fragment1pfe.csv'
    fragment2_file = 'fragment2pfe.csv'
    fragment3_file = 'fragment3pfe.csv'

    # Création du cipher AES
    cipher = AES.new(AES_KEY, AES_MODE)

    # 1) Lire le fichier original et chiffrer complètement chaque ligne, 
    # puis enregistrer dans full_encrypted_file
    with open(input_file, 'r', newline='') as csv_in, \
         open(full_encrypted_file, 'w', newline='') as csv_out:
        reader = csv.reader(csv_in)
        writer = csv.writer(csv_out)

        for row in reader:
            # Joindre la ligne en une string séparée par des virgules
            line_str = ",".join(row)
            padded_data = pad(line_str.encode(), AES.block_size)
            encrypted_data = cipher.encrypt(padded_data)
            encrypted_b64 = base64.b64encode(encrypted_data).decode()
            writer.writerow([encrypted_b64])

    # 2) Ensuite, fragmentation comme avant sur classified_output.csv
    with open(fragment1_file, 'w', newline='') as f1, \
         open(fragment2_file, 'w', newline='') as f2, \
         open(fragment3_file, 'w', newline='') as f3, \
         open(input_file, 'r', newline='') as csvfile:

        reader = csv.reader(csvfile)
        writer1 = csv.writer(f1)
        writer2 = csv.writer(f2)
        writer3 = csv.writer(f3)

        for row in reader:
            if len(row) != 3:
                continue

            part1, part2, part3 = row[0], row[1], row[2]
            if part3.strip() == '1':
                padded_data = pad(part2.encode(), AES.block_size)
                encrypted_data = cipher.encrypt(padded_data)
                encrypted_b64 = base64.b64encode(encrypted_data).decode()

                # Fragmentation en 3 morceaux
                third = len(encrypted_b64) // 3
                frag1 = encrypted_b64[:third]
                frag2 = encrypted_b64[third:2*third]
                frag3 = encrypted_b64[2*third:]

                writer1.writerow([f"{user_id},{part1},{frag1}"])
                writer2.writerow([frag2])
                writer3.writerow([frag3])

    # Fonction d'upload vers S3
    def upload_to_s3(file_name, bucket, object_name=None):
        if object_name is None:
            object_name = file_name
        try:
            s3.upload_file(file_name, bucket, object_name)
            print(f"✅ {file_name} uploadé avec succès vers {bucket}")
        except Exception as e:
            print(f"❌ Échec de l'upload de {file_name} vers {bucket} : {e}")

    # Upload fichier complet chiffré dans bucket redpfe
    upload_to_s3(full_encrypted_file, buckets['redpfe'])

    # Upload des fragments
    upload_to_s3(fragment1_file, buckets['frag1'])
    upload_to_s3(fragment2_file, buckets['frag2'])
    upload_to_s3(fragment3_file, buckets['frag3'])

# --- Load the pre-trained model and label encoder ---
clf = joblib.load('rf_sensitivity_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

def run_ai_task_on_csv(csv_file_path):
    try:
        df_to_classify = pd.read_csv(csv_file_path, encoding='ISO-8859-1', sep=';')
        if 'label' not in df_to_classify.columns:
            raise ValueError("The CSV file must contain a 'label' column")
        
        X_to_classify = df_to_classify['label'].values.reshape(-1, 1)
        X_encoded_to_classify = label_encoder.transform(X_to_classify.ravel()).reshape(-1, 1)
        y_proba = clf.predict_proba(X_encoded_to_classify)[:, 1]
        y_pred = (y_proba >= 0.7).astype(int)
        df_to_classify['Sensibilite predit'] = y_pred

        classified_df = df_to_classify[['label', 'value', 'Sensibilite predit']]
        classified_df = classified_df.sort_values(by='label')

        classified_df.to_csv('classified_output.csv', index=False, sep=',', encoding='utf-8')
        print("AI task completed. Classified file saved at: classified_output.csv")

        # Now run the encryption and upload
        results = process_csv('classified_output.csv')
        print("Encryption and upload completed:", results)

    except Exception as e:
        print(f"Error in AI task or encryption: {str(e)}")


@app.route('/connect_db', methods=['POST'])
def connect_db():
    try:
        host = request.form['host']
        user = request.form['user']
        password = request.form['password']
        database = request.form['database']

        # Création du moteur SQLAlchemy
        engine = create_engine(f'mysql+pymysql://{user}:{password}@{host}/{database}')

        # Récupération des noms de tables
        tables_query = "SHOW TABLES"
        tables_df = pd.read_sql(tables_query, engine)
        
        # Vérifier qu'on a au moins une table
        if tables_df.empty:
            return jsonify({"error": "Aucune table trouvée dans la base de données."})
        
        # Prendre le premier nom de table
        first_table = tables_df.iloc[0, 0]

        # Correction ici : Ajout de backticks autour du nom de table
        query = f"SELECT * FROM `{first_table}`"
        df = pd.read_sql(query, engine)

        # Transformation des données
        data_transformed = []
        for col in df.columns:
            for _, row in df.iterrows():
                data_transformed.append({
                    'label': col,
                    'value': row[col]
                })

        transformed_df = pd.DataFrame(data_transformed)
        csv_file_path = 'output.csv'
        transformed_df.to_csv(csv_file_path, index=False, sep=';')

        # Exécution IA + chiffrement en arrière-plan
        threading.Thread(target=run_ai_task_on_csv, args=(csv_file_path,)).start()

        return jsonify({"message": "Le fichier CSV a été généré avec succès et l'IA + chiffrement sont en cours d'exécution en arrière-plan !"})

    except Exception as e:
        return jsonify({"error": str(e)})


AES_KEY = b'ThisIsASecretKey1234567890123456'
AES_MODE = AES.MODE_ECB

s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

BUCKET_REDPFE = 'redpfe'
BUCKET_FRAG1 = 'fragment1pfe'
BUCKET_FRAG2 = 'fragment2pfe'
BUCKET_FRAG3 = 'fragment3pfe'

def download_s3_object_content(bucket_name, key):
    try:
        response = s3.get_object(Bucket=bucket_name, Key=key)
        return response['Body'].read().decode('utf-8')
    except s3.exceptions.NoSuchKey:
        print(f"Error: S3 Object Not Found - Object '{key}' not found in bucket '{bucket_name}'.")
        return None
    except s3.exceptions.ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        if error_code == 'AccessDenied':
            print(f"Error: S3 Access Denied - Check permissions for bucket '{bucket_name}' and object '{key}'.")
        else:
            print(f"Error: S3 Client Error - An AWS client error occurred while downloading '{key}' from '{bucket_name}': {e}")
        return None
    except Exception as e:
        print(f"Error: Unknown Download Error - An unexpected error occurred during download of '{key}' from '{bucket_name}': {e}")
        return None

def authenticate_and_reconstruct(input_password):
    found_user_id = None
    user_data_row_index = -1

    print("Tentative d'authentification et de reconstruction...")

    try:
        response = s3.list_objects_v2(Bucket=BUCKET_REDPFE)
        if 'Contents' not in response:
            return "Aucun fichier d'enregistrement trouvé dans le bucket redpfe.", 404

        for obj in response['Contents']:
            file_key = obj['Key']
            if file_key.endswith('.csv'):
                print(f"Analyse du fichier d'enregistrement : {file_key}")
                csv_content = download_s3_object_content(BUCKET_REDPFE, file_key)
                if csv_content:
                    reader = csv.reader(StringIO(csv_content))
                    header = next(reader, None)
                    if not header or 'ID' not in header or 'Password' not in header:
                        continue

                    id_index = header.index('ID')
                    pwd_index = header.index('Password')

                    for row in reader:
                        if len(row) > max(id_index, pwd_index):
                            if row[pwd_index] == input_password:
                                found_user_id = row[id_index]
                                print(f"Authentification réussie pour l'ID : {found_user_id}")
                                break
                    if found_user_id:
                        break
    except Exception as e:
        return f"Erreur lors de l'accès au bucket redpfe : {e}", 500

    if not found_user_id:
        return "Mot de passe incorrect ou utilisateur non trouvé.", 401

    # ========== Récupération des fragments ==========

    try:
        # Fragment 1
        content1 = download_s3_object_content(BUCKET_FRAG1, 'fragment1pfe.csv')
        if not content1:
            return "Erreur : fragment1pfe.csv introuvable.", 500

        frag1_data = target_part1 = None
        reader1 = csv.reader(StringIO(content1))
        for idx, row in enumerate(reader1):
            if len(row) > 0 and row[0].startswith(found_user_id):
                parts = row[0].split(',')
                if len(parts) >= 3 and parts[0] == found_user_id:
                    frag1_data = parts[2]
                    target_part1 = parts[1]
                    user_data_row_index = idx
                    print(f"Fragment 1 trouvé : {frag1_data[:20]}...")
                    break

        if not frag1_data:
            return "Fragment 1 non trouvé.", 404

        # Fragment 2
        content2 = download_s3_object_content(BUCKET_FRAG2, 'fragment2pfe.csv')
        if not content2:
            return "Erreur : fragment2pfe.csv introuvable.", 500

        reader2 = csv.reader(StringIO(content2))
        frag2_data = next((row[0] for idx, row in enumerate(reader2) if idx == user_data_row_index and len(row) > 0), None)

        if not frag2_data:
            return "Fragment 2 non trouvé.", 404
        print(f"Fragment 2 trouvé : {frag2_data[:20]}...")

        # Fragment 3
        content3 = download_s3_object_content(BUCKET_FRAG3, 'fragment3pfe.csv')
        if not content3:
            return "Erreur : fragment3pfe.csv introuvable.", 500

        reader3 = csv.reader(StringIO(content3))
        frag3_data = next((row[0] for idx, row in enumerate(reader3) if idx == user_data_row_index and len(row) > 0), None)

        if not frag3_data:
            return "Fragment 3 non trouvé.", 404
        print(f"Fragment 3 trouvé : {frag3_data[:20]}...")

    except Exception as e:
        return f"Erreur lors de la récupération des fragments : {e}", 500

    # ========== Déchiffrement AES ECB ==========

      # Phase 3: Reconstruction et déchiffrement des données
    encrypted_b64_full = frag1_data + frag2_data + frag3_data 
    print(f"DEBUG DÉCHIFFREMENT FINAL: Base64 RECONSTRUITE: '{encrypted_b64_full}'")
    print(f"DEBUG DÉCHIFFREMENT FINAL: Longueur de la reconstruite Base64: {len(encrypted_b64_full)}")

    try:
        encrypted_bytes = base64.b64decode(encrypted_b64_full)
        print("DEBUG: Données Base64 décodées avec succès.")
    except base64.binascii.Error as e:
        return f"Erreur de décodage Base64 : La chaîne Base64 reconstituée est invalide ou corrompue. Détails: {e}", 500
    except Exception as e:
        return f"Erreur inattendue lors du décodage Base64 des fragments: {e}", 500

    try:
        print(f"DEBUG Déchiffrement: AES_KEY: {AES_KEY}")
        print(f"DEBUG Déchiffrement: AES_MODE: {AES_MODE}")
        print(f"DEBUG Déchiffrement: Longueur des données chiffrées (bytes): {len(encrypted_bytes)}")

        cipher = AES.new(AES_KEY, AES_MODE)
        decrypted_padded_bytes = cipher.decrypt(encrypted_bytes)
        original_data = unpad(decrypted_padded_bytes, AES.block_size).decode('utf-8')
        print("DEBUG: Déchiffrement AES réussi.")
        return {
            'user_id': found_user_id,
            'target_part1': target_part1,
            'original_data': original_data
        }, 200
    except ValueError as ve:
        return f"Erreur de déchiffrement (Padding) : Les données chiffrées sont corrompues, ou la clé/le mode AES est incorrect. Détails: {ve}", 500
    except Exception as e:
        return f"Erreur inattendue lors du déchiffrement AES: {e}", 500


@app.route('/reconstruct', methods=['POST'])
def reconstruct_data():
    data = request.get_json()
    if not data or 'password' not in data:
        return jsonify({"error": "Mot de passe requis."}), 400

    input_password = data['password']
    result, status_code = authenticate_and_reconstruct(input_password)

    if status_code == 200:
        return jsonify({"message": result}), 200
    else:
        return jsonify({"error": result}), status_code

# --- The rest of your existing routes remain unchanged ---

@app.route('/acc')
def accueil():
    return render_template('acc.html')

@app.route('/accd')
def home():
    user = request.args.get('user', 'Utilisateur')
    return render_template('accd.html', user=user)

@app.route('/signup')
def makecompte():
    return render_template('makecompte.html')

@app.route('/inscri')
def inscrit():
    return render_template('inscri.html')

@app.route('/login_face', methods=['POST'])
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

@app.route('/acceder')
def acceder():
    return render_template('rec.html')

@app.route('/face', methods=['POST'])
def save_face_image():
    try:
        data = request.get_json()
        name = data.get('name')
        image_data = data.get('image_data')
        action = data.get('action')

        if not name or not image_data:
            return "Nom ou image manquant.", 400

        encoded = image_data.split(",")[1]
        img_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        user_dir = os.path.join("images", name)
        os.makedirs(user_dir, exist_ok=True)
        count = len(os.listdir(user_dir))
        filename = f"{count+1}.jpg"
        cv2.imwrite(os.path.join(user_dir, filename), img)

        if action == "enregistrer":
            return jsonify({"success": True, "message": "Image enregistrée"})
        else:
            return jsonify({"success": False, "message": "Action non reconnue"})

    except Exception as e:
        return jsonify({"success": False, "message": "Erreur lors de l'enregistrement : " + str(e)})

if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5000/acc")
    app.run(debug=True)