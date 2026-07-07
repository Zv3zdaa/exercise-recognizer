import pandas as pd
import joblib
import os
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks.python import vision
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from crud import extract_angles, clean_df, extract_window_features

mp_pose = mp.solutions.pose
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker_full.task'),
    running_mode=mp.tasks.vision.RunningMode.IMAGE)


dt = []
entries = os.listdir("./data")
with PoseLandmarker.create_from_options(options) as landmarker:
    for i in entries:
        for x in os.listdir(f"./data/{i}"):
            cap = cv.VideoCapture(f'./data/{i}/{x}')
            fps = cap.get(cv.CAP_PROP_FPS); step = max(1, round(fps/25))
            total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
            for frame_count in range(total_frames):
                ret, frame = cap.read()
                if not ret or frame_count < 10 or frame_count > total_frames-10 or frame_count % step != 0:
                    continue
                frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                result = landmarker.detect(mp_image)
                if result.pose_landmarks:
                    angles = extract_angles(result.pose_landmarks[0])
                    dt.append({'video': x, 'file': f'{x}_{frame_count:04d}', 'exercise': i, **angles})
            cap.release()
df = pd.DataFrame(dt); df.to_csv('data.csv', index=False)

df = pd.read_csv('data.csv')
df_clean = clean_df(df)

window_features = df_clean.groupby(['video','window_id']).apply(extract_window_features).reset_index()
window_y = df_clean.groupby(['video','window_id'])['exercise'].agg(lambda x: x.mode().iloc[0])
window_data = window_features.merge(window_y, on=['video','window_id'])

feature_cols = [c for c in window_data.columns if c not in ['video','window_id','exercise']]
X, y = window_data[feature_cols], window_data['exercise']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=1000, max_depth=7, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)
joblib.dump((model, feature_cols), 'exercise_model.pkl')
print("Model saved.")