# app_functions.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import insightface

def load_models(age_model_path, gender_model_path, face_model_name='buffalo_l'):
    age_model = load_model(age_model_path)
    gender_model = load_model(gender_model_path)
    face_model = insightface.app.FaceAnalysis(name=face_model_name, providers=['CPUExecutionProvider'])
    face_model.prepare(ctx_id=0)
    return age_model, gender_model, face_model

def extract_and_preprocess_faces(img_path, face_model):
    img = cv2.imread(img_path)
    faces = face_model.get(img)
    if not faces:
        raise ValueError(f"No face detected in image: {img_path}")
    face = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]), reverse=True)[0]
    x1, y1, x2, y2 = map(int, face.bbox)
    face_crop = img[y1:y2, x1:x2]
    face_resized = cv2.resize(face_crop, (112, 112))
    face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    return face, face_gray

def predict_age(face_img_gray, age_model):
    img = face_img_gray.astype("float32") / 255.0
    img = np.expand_dims(img, axis=(0, -1))
    predicted_age = age_model.predict(img)[0][0]
    return predicted_age

def predict_gender(face_img_gray, gender_model):
    img = face_img_gray.astype("float32") / 255.0
    img = np.expand_dims(img, axis=(0, -1))
    pred_prob = gender_model.predict(img)[0][0]
    gender = "Male" if pred_prob < 0.5 else "Female"
    return gender, pred_prob

def match_faces(face1, face2, criteria_thresh=1, cos_thresh=0.4, eucl_thresh=1.0, angle_thresh=50):
    def angular_distance(v1, v2):
        cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.arccos(np.clip(cos_sim, -1.0, 1.0)) * (180 / np.pi)

    emb1 = face1.embedding / np.linalg.norm(face1.embedding)
    emb2 = face2.embedding / np.linalg.norm(face2.embedding)

    cos_sim = np.dot(emb1, emb2)
    eucl_dist = np.linalg.norm(emb1 - emb2)
    angle = angular_distance(emb1, emb2)

    votes = sum([
        cos_sim > cos_thresh,
        eucl_dist < eucl_thresh,
        angle < angle_thresh
    ])
    is_match = votes >= criteria_thresh

    return {
        "cos_sim": cos_sim,
        "eucl_dist": eucl_dist,
        "angle": angle,
        "is_match": is_match,
        "votes": votes
    }
