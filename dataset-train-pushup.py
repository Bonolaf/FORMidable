import os
import csv
from datetime import datetime

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_DIR = "D:\ESAIP\ING2 25 - 26\Projet Perso\Dataset FORMidable\Pushup"
MP_MODEL_PATH = os.path.join("models", "pose_landmarker_full.task")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def build_progress_message(current, total, prefix="Progress", width=30):
    if total is None or total <= 0:
        return f"{prefix}: {current} frames processed"

    percent = min(100, int((current / total) * 100))
    filled = int((width * percent) / 100)
    bar = "#" * filled + "-" * (width - filled)
    return f"{prefix}: |{bar}| {percent:3d}% ({current}/{total})"


def print_progress(current, total, prefix="Progress", width=30):
    if total is None:
        if current == 1:
            print(build_progress_message(current, total, prefix, width), end="\r", flush=True)
        return

    step = max(1, total // 10)
    if current == 1 or current == total or current % step == 0:
        print(build_progress_message(current, total, prefix, width), end="\r", flush=True)


def finish_progress():
    print()


def landmark_xy(landmarks, idx):
    lm = landmarks[idx]
    return np.array([lm.x, lm.y], dtype=np.float32)


def angle(a, b, c):
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
    cosang = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def point_to_line_distance(point, line_p1, line_p2):
    x0, y0 = point
    x1, y1 = line_p1
    x2, y2 = line_p2

    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2

    return abs(A * x0 + B * y0 + C) / (A**2 + B**2) ** 0.5


def point_to_point_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def pushup_elbow_average(landmarks):
    LS, LE, LW = 11, 13, 15
    RS, RE, RW = 12, 14, 16

    l_sh = landmark_xy(landmarks, LS)
    l_el = landmark_xy(landmarks, LE)
    l_wr = landmark_xy(landmarks, LW)
    r_sh = landmark_xy(landmarks, RS)
    r_el = landmark_xy(landmarks, RE)
    r_wr = landmark_xy(landmarks, RW)

    elbow_l = angle(l_sh, l_el, l_wr)
    elbow_r = angle(r_sh, r_el, r_wr)

    return (elbow_l + elbow_r) / 2.0


def is_pushup_contact_pose(landmarks):
    LS, RS = 11, 12
    LW, RW = 15, 16

    l_sh = landmark_xy(landmarks, LS)
    r_sh = landmark_xy(landmarks, RS)
    l_wr = landmark_xy(landmarks, LW)
    r_wr = landmark_xy(landmarks, RW)

    return (l_wr[1] > l_sh[1] + 0.05) and (r_wr[1] > r_sh[1] + 0.05)


def extract_features(landmarks):
    LE, RE = 7, 8
    LS, RS = 11, 12
    LH, RH = 23, 24
    LK, RK = 25, 26
    LA, RA = 27, 28

    l_ear = landmark_xy(landmarks, LE)
    r_ear = landmark_xy(landmarks, RE)
    l_sh = landmark_xy(landmarks, LS)
    r_sh = landmark_xy(landmarks, RS)
    l_hp = landmark_xy(landmarks, LH)
    r_hp = landmark_xy(landmarks, RH)
    l_kn = landmark_xy(landmarks, LK)
    r_kn = landmark_xy(landmarks, RK)
    l_an = landmark_xy(landmarks, LA)
    r_an = landmark_xy(landmarks, RA)

    l_hip_to_line = point_to_line_distance(l_hp, l_sh, l_an)
    r_hip_to_line = point_to_line_distance(r_hp, r_sh, r_an)
    l_knee_to_line = point_to_line_distance(l_kn, l_sh, l_an)
    r_knee_to_line = point_to_line_distance(r_kn, r_sh, r_an)
    l_ear_to_l_sh = point_to_point_distance(l_ear, l_sh)
    r_ear_to_r_sh = point_to_point_distance(r_ear, r_sh)

    return [
        l_hip_to_line,
        r_hip_to_line,
        l_knee_to_line,
        r_knee_to_line,
        l_ear_to_l_sh,
        r_ear_to_r_sh,
    ]


def create_landmarker():
    base_options = python.BaseOptions(model_asset_path=MP_MODEL_PATH)

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )

    return vision.PoseLandmarker.create_from_options(options)


def find_videos(folder):
    if not os.path.exists(folder):
        return []

    return sorted(
        [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(VIDEO_EXTENSIONS)
        ]
    )


def extract_video_features(video_path, label, landmarker, timestamp_offset_ms=0):
    rows = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Impossible d'ouvrir la video: {video_path}")
        return rows

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    frame_index = 0
    rep_stage = "up"
    active_rep = False
    rep_last_landmarks = None
    rep_last_timestamp = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = timestamp_offset_ms + int((frame_index / fps) * 1000)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            landmarks = result.pose_landmarks[0]
            elbow_avg = pushup_elbow_average(landmarks)
            pushup_contact = is_pushup_contact_pose(landmarks)

            if pushup_contact and rep_stage == "up" and elbow_avg < 90:
                rep_stage = "down"
                active_rep = True
                rep_last_landmarks = landmarks
                rep_last_timestamp = timestamp_ms

            elif active_rep:
                rep_last_landmarks = landmarks
                rep_last_timestamp = timestamp_ms

                if pushup_contact and elbow_avg > 150:
                    feats = extract_features(rep_last_landmarks)
                    rows.append(
                        [
                            datetime.utcnow().isoformat(),
                            "pushup",
                            os.path.basename(video_path),
                            frame_index,
                            rep_last_timestamp,
                            *feats,
                            label,
                        ]
                    )
                    active_rep = False
                    rep_stage = "up"
                    rep_last_landmarks = None

        frame_index += 1
        print_progress(frame_index, total_frames, prefix=f"{os.path.basename(video_path)}")

    cap.release()
    finish_progress()
    print(f"{os.path.basename(video_path)} -> {len(rows)} reps utilises")

    return rows


def build_dataset():
    out_csv = "dataset_pushup.csv"

    header = [
        "timestamp",
        "exercise",
        "video_name",
        "frame_index",
        "timestamp_ms",
        "l_hip_to_line",
        "r_hip_to_line",
        "l_knee_to_line",
        "r_knee_to_line",
        "l_ear_to_l_sh",
        "r_ear_to_r_sh",
        "label",
    ]

    all_rows = []
    timestamp_offset_ms = 0

    good_dir = os.path.join(DATA_DIR, "good")
    bad_dir = os.path.join(DATA_DIR, "bad")

    if not os.path.exists(good_dir):
        print(f"Dossier manquant: {good_dir}")
        return None

    if not os.path.exists(bad_dir):
        print(f"Dossier manquant: {bad_dir}")
        return None

    with create_landmarker() as landmarker:
        for label_name, label_value in [("good", 1), ("bad", 0)]:
            label_dir = os.path.join(DATA_DIR, label_name)
            videos = find_videos(label_dir)

            print(f"\nPushup/{label_name}: {len(videos)} videos trouvees")

            for video_path in videos:
                rows = extract_video_features(
                    video_path=video_path,
                    label=label_value,
                    landmarker=landmarker,
                    timestamp_offset_ms=timestamp_offset_ms,
                )

                all_rows.extend(rows)
                timestamp_offset_ms += 10_000_000

    if len(all_rows) == 0:
        print("Aucune donn?e extraite pour le pushup")
        return None

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_rows)

    print(f"\nDataset cree: {out_csv}")
    print(f"Nombre total de lignes: {len(all_rows)}")

    return out_csv


def train_model(dataset_csv):
    if not os.path.exists(dataset_csv):
        print(f"Dataset introuvable: {dataset_csv}")
        return

    df = pd.read_csv(dataset_csv)

    feature_cols = [
        "frame_index",
        "l_hip_to_line",
        "r_hip_to_line",
        "l_knee_to_line",
        "r_knee_to_line",
        "l_ear_to_l_sh",
        "r_ear_to_r_sh",
    ]

    X = df[feature_cols].values
    y = df["label"].values

    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        print("\nPushup: entra�nement good/bad impossible.")
        print("Il n'y a qu'une seule classe dans le dataset.")
        return

    if len(df) < 10:
        print("Pushup: pas assez de donnees.")
        return

    counts = df["label"].value_counts()
    print("\nRepartition des labels:")
    print(counts)

    stratify_arg = y if counts.min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_arg,
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n=== Resultats entrainement pushup ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    out_model = "model_pushup.pkl"
    joblib.dump(model, out_model)
    print(f"Modele sauvegarde: {out_model}")


def main():
    if not os.path.exists(DATA_DIR):
        print(f"Dossier data introuvable: {DATA_DIR}")
        return

    if not os.path.exists(MP_MODEL_PATH):
        print(f"Modele MediaPipe introuvable: {MP_MODEL_PATH}")
        return

    dataset_csv = build_dataset()
    if dataset_csv:
        train_model(dataset_csv)

if __name__ == "__main__":
    main()
