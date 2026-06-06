import os
import cv2
import joblib
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import math


MP_MODEL_PATH = os.path.join("models", "pose_landmarker_full.task")

MODELS = {
    "dip": "model_dip.pkl",
    "pushup": "model_pushup.pkl",
    "squat": "model_squat.pkl",
}


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

    # Line coefficients
    A = y2 - y1
    B = x1 - x2
    C = x2*y1 - x1*y2

    # Distance formula
    distance = abs(A*x0 + B*y0 + C) / math.sqrt(A**2 + B**2)

    return distance

def point_to_point_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def line_y_at_x(line_p1, line_p2, x):
    x1, y1 = line_p1
    x2, y2 = line_p2
    if abs(x2 - x1) < 1e-6:
        return (y1 + y2) / 2.0
    return y1 + (x - x1) * (y2 - y1) / (x2 - x1)


def body_is_horizontal(landmarks, max_deviation=0.06):
    LS, RS = 11, 12
    LH, RH = 23, 24
    LK, RK = 25, 26
    LA, RA = 27, 28

    l_sh = landmark_xy(landmarks, LS)
    r_sh = landmark_xy(landmarks, RS)
    l_hp = landmark_xy(landmarks, LH)
    r_hp = landmark_xy(landmarks, RH)
    l_kn = landmark_xy(landmarks, LK)
    r_kn = landmark_xy(landmarks, RK)
    l_an = landmark_xy(landmarks, LA)
    r_an = landmark_xy(landmarks, RA)

    shoulder_center = (l_sh + r_sh) / 2.0
    ankle_center = (l_an + r_an) / 2.0
    hip_center = (l_hp + r_hp) / 2.0
    knee_center = (l_kn + r_kn) / 2.0

    hip_deviation = point_to_line_distance(hip_center, shoulder_center, ankle_center)
    knee_deviation = point_to_line_distance(knee_center, shoulder_center, ankle_center)

    return hip_deviation < max_deviation and knee_deviation < max_deviation


def is_pushup_contact_pose(landmarks):
    LS, RS = 11, 12
    LW, RW = 15, 16

    l_sh = landmark_xy(landmarks, LS)
    r_sh = landmark_xy(landmarks, RS)
    l_wr = landmark_xy(landmarks, LW)
    r_wr = landmark_xy(landmarks, RW)

    wrists_below_shoulders = (l_wr[1] > l_sh[1] + 0.05) and (r_wr[1] > r_sh[1] + 0.05)
    return wrists_below_shoulders and body_is_horizontal(landmarks)


def compute_pushup_rep_feedback(landmarks, angles):
    LS, RS = 11, 12
    LH, RH = 23, 24
    LK, RK = 25, 26
    LA, RA = 27, 28
    LE, RE = 7, 8

    l_sh = landmark_xy(landmarks, LS)
    r_sh = landmark_xy(landmarks, RS)
    l_hp = landmark_xy(landmarks, LH)
    r_hp = landmark_xy(landmarks, RH)
    l_kn = landmark_xy(landmarks, LK)
    r_kn = landmark_xy(landmarks, RK)
    l_an = landmark_xy(landmarks, LA)
    r_an = landmark_xy(landmarks, RA)
    l_ear = landmark_xy(landmarks, LE)
    r_ear = landmark_xy(landmarks, RE)

    l_ear_to_l_sh = point_to_point_distance(l_ear, l_sh)
    r_ear_to_r_sh = point_to_point_distance(r_ear, r_sh)
    shoulder_width = max(point_to_point_distance(l_sh, r_sh), 1e-6)
    avg_ear_sh = (l_ear_to_l_sh + r_ear_to_r_sh) / 2.0
    ear_ratio = avg_ear_sh / shoulder_width

    feedback = []
    if ear_ratio < 0.22:
        feedback.append("Keep your ear away from your shoulder")

    left_hip_to_line = point_to_line_distance(l_hp, l_sh, l_an)
    right_hip_to_line = point_to_line_distance(r_hp, r_sh, r_an)
    left_knee_to_line = point_to_line_distance(l_kn, l_sh, l_an)
    right_knee_to_line = point_to_line_distance(r_kn, r_sh, r_an)

    hip_dist = (left_hip_to_line + right_hip_to_line) / 2.0
    knee_dist = (left_knee_to_line + right_knee_to_line) / 2.0

    if hip_dist > 0.6 or knee_dist > 0.6:
        hip_y = (l_hp[1] + r_hp[1]) / 2.0
        line_y_left = line_y_at_x(l_sh, l_an, l_hp[0])
        line_y_right = line_y_at_x(r_sh, r_an, r_hp[0])
        avg_line_y = (line_y_left + line_y_right) / 2.0

        if hip_y > avg_line_y:
            feedback.append("Raise your hips to make a straight line")
        else:
            feedback.append("Lower your hips to make a straight line")

    if not feedback:
        feedback.append("Good pushup form")

    return feedback


def extract_features(landmarks):
    L_ear, R_ear = 7, 8
    LS, RS = 11, 12
    LE, RE = 13, 14
    LW, RW = 15, 16
    LH, RH = 23, 24
    LK, RK = 25, 26
    LA, RA = 27, 28

    l_ear = landmark_xy(landmarks, L_ear)
    r_ear = landmark_xy(landmarks, R_ear)
    l_sh = landmark_xy(landmarks, LS)
    r_sh = landmark_xy(landmarks, RS)
    l_el = landmark_xy(landmarks, LE)
    r_el = landmark_xy(landmarks, RE)
    l_wr = landmark_xy(landmarks, LW)
    r_wr = landmark_xy(landmarks, RW)
    l_hp = landmark_xy(landmarks, LH)
    r_hp = landmark_xy(landmarks, RH)
    l_kn = landmark_xy(landmarks, LK)
    r_kn = landmark_xy(landmarks, RK)
    l_an = landmark_xy(landmarks, LA)
    r_an = landmark_xy(landmarks, RA)

    elbow_l = angle(l_sh, l_el, l_wr)
    elbow_r = angle(r_sh, r_el, r_wr)

    knee_l = angle(l_hp, l_kn, l_an)
    knee_r = angle(r_hp, r_kn, r_an)

    hip_l = angle(l_sh, l_hp, l_kn)
    hip_r = angle(r_sh, r_hp, r_kn)

    shoulder_l = angle(l_el, l_sh, l_hp)
    shoulder_r = angle(r_el, r_sh, r_hp)

    diff_elbow = abs(elbow_l - elbow_r)
    diff_knee = abs(knee_l - knee_r)
    diff_hip = abs(hip_l - hip_r)

    l_hip_to_line = point_to_line_distance(l_hp, l_sh, l_an)
    r_hip_to_line = point_to_line_distance(r_hp, r_sh, r_an)
    l_knee_to_line = point_to_line_distance(l_kn, l_sh, l_an)
    r_knee_to_line = point_to_line_distance(r_kn, r_sh, r_an)
    l_ear_to_l_sh = point_to_point_distance(l_ear, l_sh)
    r_ear_to_r_sh = point_to_point_distance(r_ear, r_sh)

    features = [
        l_hip_to_line,
        r_hip_to_line,
        l_knee_to_line,
        r_knee_to_line,
        l_ear_to_l_sh,
        r_ear_to_r_sh
    ]

    angles_dict = {
        "elbow_l": elbow_l,
        "elbow_r": elbow_r,
        "knee_l": knee_l,
        "knee_r": knee_r,
        "hip_l": hip_l,
        "hip_r": hip_r,
        "shoulder_l": shoulder_l,
        "shoulder_r": shoulder_r,
        "diff_elbow": diff_elbow,
        "diff_knee": diff_knee,
        "diff_hip": diff_hip,
    }

    return features, angles_dict


def math_check(exercise, angles):
    problems = []
    score = 0

    elbow_avg = (angles["elbow_l"] + angles["elbow_r"]) / 2
    knee_avg = (angles["knee_l"] + angles["knee_r"]) / 2
    hip_avg = (angles["hip_l"] + angles["hip_r"]) / 2

    if exercise == "pushup":
        if elbow_avg < 70:
            score += 1
        else:
            problems.append("Descends plus bas")

        if angles["diff_elbow"] < 25:
            score += 1
        else:
            problems.append("Bras pas symetriques")

        if hip_avg > 130:
            score += 1
        else:
            problems.append("Garde le corps plus droit")

    elif exercise == "squat":
        if knee_avg < 110:
            score += 1
        else:
            problems.append("Descends plus bas")

        if angles["diff_knee"] < 25:
            score += 1
        else:
            problems.append("Genoux pas symetriques")

        if hip_avg < 130:
            score += 1
        else:
            problems.append("Plie plus les hanches")

    elif exercise == "dip":
        if elbow_avg < 100:
            score += 1
        else:
            problems.append("Descends plus bas")

        if angles["diff_elbow"] < 25:
            score += 1
        else:
            problems.append("Bras pas symetriques")

        if hip_avg > 90:
            score += 1
        else:
            problems.append("Stabilise ton corps")

    math_good = score >= 2

    return math_good, problems


def draw_pushup_overlay(frame, landmarks, angles):
    h, w, _ = frame.shape

    LS, RS = 11, 12
    LH, RH = 23, 24
    LK, RK = 25, 26
    LA, RA = 27, 28

    l_sh = landmarks[LS]
    r_sh = landmarks[RS]
    l_hp = landmarks[LH]
    r_hp = landmarks[RH]
    l_kn = landmarks[LK]
    r_kn = landmarks[RK]
    l_an = landmarks[LA]
    r_an = landmarks[RA]

    def to_px(lm):
        return (int(lm.x * w), int(lm.y * h))

    left_shoulder = to_px(l_sh)
    right_shoulder = to_px(r_sh)
    left_hip = to_px(l_hp)
    right_hip = to_px(r_hp)
    left_knee = to_px(l_kn)
    right_knee = to_px(r_kn)
    left_ankle = to_px(l_an)
    right_ankle = to_px(r_an)

    cv2.line(frame, left_shoulder, left_ankle, (255, 255, 255), 2)
    cv2.line(frame, right_shoulder, right_ankle, (255, 255, 255), 2)

    cv2.circle(frame, left_hip, 6, (0, 255, 255), -1)
    cv2.circle(frame, right_hip, 6, (0, 255, 255), -1)
    cv2.circle(frame, left_knee, 6, (255, 255, 0), -1)
    cv2.circle(frame, right_knee, 6, (255, 255, 0), -1)

    # angle_texts = [                                   # Print elbow, knee, hip angles
    #     f"Elbow L: {angles['elbow_l']:.1f}°",
    #     f"Elbow R: {angles['elbow_r']:.1f}°",
    #     f"Knee L: {angles['knee_l']:.1f}°",
    #     f"Knee R: {angles['knee_r']:.1f}°",
    #     f"Hip L: {angles['hip_l']:.1f}°",
    #     f"Hip R: {angles['hip_r']:.1f}°",
    # ]

    # y_offset = 150
    # for text in angle_texts:
    #     cv2.putText(frame, text, (10, y_offset),
    #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    #     y_offset += 25


def create_landmarker():
    base_options = python.BaseOptions(model_asset_path=MP_MODEL_PATH)

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )

    return vision.PoseLandmarker.create_from_options(options)


def predict_camera(exercise="pushup", camera_id=0):
    if exercise not in MODELS:
        print("Exercice invalide.")
        print("Choisis parmi:", list(MODELS.keys()))
        return

    model_path = MODELS[exercise]

    if not os.path.exists(model_path):
        print(f"Modèle introuvable: {model_path}")
        return

    if not os.path.exists(MP_MODEL_PATH):
        print(f"Modèle MediaPipe introuvable: {MP_MODEL_PATH}")
        return

    model = joblib.load(model_path)

    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print("Impossible d'ouvrir la caméra.")
        return

    print("Caméra lancée.")
    print("Appuie sur Q pour quitter.")

    frame_index = 0
    fps = 30
    rep_stage = "up"
    rep_count = 0
    good_rep_count = 0
    bad_rep_count = 0
    sum_ai_confidence = 0.0
    last_rep_feedback = ""
    feedback_frames_left = 0
    last_ai_confidence = 0.0
    last_final_result = "WAITING"
    last_color = (0, 0, 255)

    with create_landmarker() as landmarker:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame_rgb,
            )

            timestamp_ms = int((frame_index / fps) * 1000)

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms,
            )

            label_text = "Aucune pose detectee"
            advice_text = ""
            color = (0, 0, 255)

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                landmarks = result.pose_landmarks[0]
                features, angles = extract_features(landmarks)

                draw_pushup_overlay(frame, landmarks, angles)

                # Maths
                math_good, problems = math_check(exercise, angles)

                if exercise == "pushup":
                    elbow_avg = (angles["elbow_l"] + angles["elbow_r"]) / 2.0
                    pushup_contact = is_pushup_contact_pose(landmarks)

                    if pushup_contact and rep_stage == "up" and elbow_avg < 90:
                        rep_stage = "down"

                    if pushup_contact and rep_stage == "down" and elbow_avg > 150:
                        rep_stage = "up"
                        rep_count += 1

                        X = np.array([frame_index, *features], dtype=np.float32).reshape(1, -1)
                        ai_pred = model.predict(X)[0]

                        confidence = 0.0
                        if hasattr(model, "predict_proba"):
                            proba = model.predict_proba(X)[0]
                            confidence = float(np.max(proba))

                        last_ai_confidence = confidence
                        sum_ai_confidence += confidence

                        if ai_pred == 1 and math_good:
                            last_final_result = "good"
                            last_color = (0, 255, 0)
                            good_rep_count += 1
                        elif ai_pred == 1 and not math_good:
                            last_final_result = "moyen"
                            last_color = (0, 165, 255)
                            bad_rep_count += 1
                        else:
                            last_final_result = "bad"
                            last_color = (0, 0, 255)
                            bad_rep_count += 1

                        rep_feedback = compute_pushup_rep_feedback(landmarks, angles)
                        if rep_feedback == ["Good pushup form"] and ai_pred == 1 and math_good:
                            last_rep_feedback = f"Last rep: {last_final_result} | IA: {last_ai_confidence:.2f}"
                        else:
                            last_rep_feedback = " | ".join(rep_feedback + problems)
                        feedback_frames_left = int(fps * 2)

                if rep_count == 0:
                    label_text = f"{exercise.upper()} : Waiting for rep"
                    color = (0, 0, 255)
                else:
                    label_text = f"Last rep: {last_final_result} | IA: {last_ai_confidence:.2f} | Total reps: {rep_count}"
                    color = last_color

                if feedback_frames_left > 0:
                    advice_text = last_rep_feedback
                    feedback_frames_left -= 1
                else:
                    advice_text = ""

            cv2.putText(
                frame,
                label_text,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
            )

            if advice_text:
                cv2.putText(
                    frame,
                    advice_text,
                    (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow("IA", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_index += 1

    cap.release()
    cv2.destroyAllWindows()

    print("\n=== Pushup session report ===")
    print(f"Total reps: {rep_count}")
    if rep_count > 0:
        avg_confidence = sum_ai_confidence / rep_count
        print(f"Good reps: {good_rep_count}")
        print(f"Bad reps: {bad_rep_count}")
        print(f"Last result: {last_final_result}")
        print(f"Last AI score: {last_ai_confidence:.2f}")
        print(f"Average AI score: {avg_confidence:.2f}")
    else:
        print("No reps were completed.")


if __name__ == "__main__":
    predict_camera(exercise="pushup", camera_id=0)