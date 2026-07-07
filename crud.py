import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import cv2 as cv

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(cosine))

def extract_angles(landmarks):
    def get_point(name):
        lm = landmarks[getattr(mp.tasks.vision.PoseLandmark, name)]
        if lm.visibility < 0.3:
            return None
        return (lm.x, lm.y)

    points = {name: get_point(name) for name in [
        'LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST',
        'RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST',
        'LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE',
        'RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE'
    ]}

    angles = {}
    for angle_name, (a, b, c) in {
        'left_elbow':  ('LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST'),
        'right_elbow': ('RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST'),
        'left_knee':   ('LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE'),
        'right_knee':  ('RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE')
    }.items():
        if points[a] is None or points[b] is None or points[c] is None:
            angles[angle_name] = None
        else:
            angles[angle_name] = calculate_angle(points[a], points[b], points[c])
    return angles

def clean_df(df):
    df['frame_num'] = df['file'].str.extract(r'frame_(\d+)').astype(int)
    df = df.sort_values(['video', 'frame_num'])

    angle_cols = ['left_elbow', 'right_elbow', 'left_knee', 'right_knee']
    df[angle_cols] = df.groupby('video')[angle_cols].transform(
        lambda x: x.rolling(window=5, center=True, min_periods=1).median()
    )

    for col in angle_cols:
        diff = df.groupby('video')[col].diff().abs()
        df.loc[diff > 30, col] = None
        df[col] = df.groupby('video')[col].ffill()

    mask = df['exercise'].str.contains('Pushups')
    df.loc[mask, 'left_knee'] = df.loc[mask, 'left_knee'].fillna(180)
    df.loc[mask, 'right_knee'] = df.loc[mask, 'right_knee'].fillna(180)

    df['window_id'] = df.groupby('video').cumcount() // 15
    return df.dropna()

def extract_window_features(group):
    res = {}
    for col in ['left_elbow', 'right_elbow', 'left_knee', 'right_knee']:
        vals = group[col]
        res[f'{col}_mean'] = vals.mean()
        res[f'{col}_std'] = vals.std()
        res[f'{col}_min'] = vals.min()
        res[f'{col}_max'] = vals.max()
        res[f'{col}_trend'] = vals.iloc[-1] - vals.iloc[0]
    res['knee_diff_mean'] = abs(res['left_knee_mean'] - res['right_knee_mean'])
    res['knee_diff_max'] = max(
        abs(group['left_knee'].max() - group['right_knee'].max()),
        abs(group['left_knee'].min() - group['right_knee'].min())
    )
    res['lunge_flag'] = int(
        (res['left_knee_max'] > 140 and res['right_knee_min'] < 90) or
        (res['right_knee_max'] > 140 and res['left_knee_min'] < 90)
    )
    res['knee_diff_std'] = abs(group['left_knee'] - group['right_knee']).std()
    return pd.Series(res)

def compute_features_from_buffer(buffer):
    df = pd.DataFrame(buffer)
    feats = {}
    for col in ['left_elbow', 'right_elbow', 'left_knee', 'right_knee']:
        s = df[col]
        feats[f'{col}_mean'] = s.mean()
        feats[f'{col}_std'] = s.std()
        feats[f'{col}_min'] = s.min()
        feats[f'{col}_max'] = s.max()
        feats[f'{col}_trend'] = s.iloc[-1] - s.iloc[0]
    feats['knee_diff_mean'] = abs(feats['left_knee_mean'] - feats['right_knee_mean'])
    feats['knee_diff_max'] = max(
        abs(df['left_knee'].max() - df['right_knee'].max()),
        abs(df['left_knee'].min() - df['right_knee'].min())
    )
    feats['lunge_flag'] = int(
        (feats['left_knee_max'] > 140 and feats['right_knee_min'] < 90) or
        (feats['right_knee_max'] > 140 and feats['left_knee_min'] < 90)
    )
    feats['knee_diff_std'] = abs(df['left_knee'] - df['right_knee']).std()
    return feats

def draw_pose(image_bgr, landmarks):
    h, w = image_bgr.shape[:2]
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv.circle(image_bgr, (cx, cy), 4, (0, 255, 0), -1)
    connections = [(11,13),(13,15),(12,14),(14,16),(23,25),(25,27),(24,26),(26,28),(11,12),(23,24),(11,23),(12,24)]
    for start, end in connections:
        x1, y1 = int(landmarks[start].x * w), int(landmarks[start].y * h)
        x2, y2 = int(landmarks[end].x * w), int(landmarks[end].y * h)
        cv.line(image_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return image_bgr

def count(angle_left_knee, angle_right_knee, angle_left_elbow, angle_right_elbow, state, gym):
    if any(v is None or np.isnan(v) for v in [angle_left_knee, angle_right_knee, angle_left_elbow, angle_right_elbow]):
        return state, False
    avg_knee = (angle_left_knee + angle_right_knee) / 2
    avg_elbow = (angle_left_elbow + angle_right_elbow) / 2
    if gym == 'Squats':
        if avg_knee < 90 and state == 'up': return 'down', False
        if avg_knee > 150 and state == 'down': return 'up', True
    elif gym == 'Pushups':
        if avg_elbow < 90 and state == 'up': return 'down', False
        if avg_elbow > 150 and state == 'down': return 'up', True
    return state, False