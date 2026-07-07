import time
import pandas as pd
import joblib
import os
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks.python import vision
from crud import draw_pose, extract_angles, compute_features_from_buffer, count

model, feature_cols = joblib.load('exercise_model.pkl')

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker_full.task'),
    running_mode=mp.tasks.vision.RunningMode.IMAGE)

video_path = input('Video path: ').strip()
cap = cv.VideoCapture(video_path)
fps = cap.get(cv.CAP_PROP_FPS); step = max(1, round(fps/25))
out = cv.VideoWriter('output.mp4', cv.VideoWriter_fourcc(*'mp4v'), 25, (int(cap.get(3)), int(cap.get(4))))

window_buffer = []
current_exercise = None
counters = {'Squats': 0, 'Pushups': 0, 'Plank': 0}
state = 'up'; frame_cnt = 0
plank_start = None; plank_accum = 0.0; plank_lost = 0

with PoseLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_cnt += 1
        if frame_cnt % step != 0: continue

        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb))
        if result.pose_landmarks:
            draw_pose(frame, result.pose_landmarks[0])
            angles = extract_angles(result.pose_landmarks[0])
            if not any(v is None for v in angles.values()):
                window_buffer.append(angles)

        if len(window_buffer) == 15:
            feats = compute_features_from_buffer(window_buffer)
            X = pd.DataFrame([feats])[feature_cols]
            pred = model.predict(X)[0]

            if pred == 'Plank':
                if plank_start is None: plank_start = time.time()
                plank_accum = time.time() - plank_start
                plank_lost = 0
            else:
                if plank_start is not None:
                    plank_lost += 1
                    if plank_lost >= 75:
                        print(f"Plank held {plank_accum:.1f}s")
                        plank_start = None; plank_accum = 0.0

            if 'streak' not in locals(): streak = {'ex': pred, 'cnt': 1}
            elif streak['ex'] == pred:
                streak['cnt'] += 1
                if streak['cnt'] >= 3 and pred != current_exercise:
                    current_exercise = pred
                    state = 'up'; streak = None
            else: streak = {'ex': pred, 'cnt': 1}

            last = window_buffer[-1]
            state, inc = count(last['left_knee'], last['right_knee'],
                               last['left_elbow'], last['right_elbow'], state, current_exercise)
            if inc: counters[current_exercise] += 1

            window_buffer.pop(0)

        display_ex = current_exercise if current_exercise else "Detecting..."
        cv.putText(frame, f"Exercise: {display_ex}", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)
        if current_exercise == 'Plank':
            cv.putText(frame, f"Plank: {plank_accum:.1f}s", (10, 150), cv.FONT_HERSHEY_SIMPLEX, 2, (255,255,0), 3)
        out.write(frame)

cap.release(); out.release()