from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import cv2
import mediapipe as mp

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(project_root, 'hand_landmarker.task')

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options,
    num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

'''
Thumb: 
  1: THUMB_CMC  (base of thumb)
  2: THUMB_MCP  (knuckle)
  3: THUMB_IP   (middle joint)
  4: THUMB_TIP  (fingertip)
Index:
  5: INDEX_FINGER_MCP  (knuckle/base)
  6: INDEX_FINGER_PIP  (middle joint)
  7: INDEX_FINGER_DIP  (joint near tip)
  8: INDEX_FINGER_TIP  (fingertip)
Middle:
  9: MIDDLE_FINGER_MCP  (knuckle/base)
  10: MIDDLE_FINGER_PIP (middle joint)
  11: MIDDLE_FINGER_DIP (joint near tip)
  12: MIDDLE_FINGER_TIP (fingertip)
Ring:
  13: RING_FINGER_MCP  (knuckle/base)
  14: RING_FINGER_PIP  (middle joint)
  15: RING_FINGER_DIP  (joint near tip)
  16: RING_FINGER_TIP  (fingertip)
Pinky:
  17: PINKY_MCP  (knuckle/base)
  18: PINKY_PIP  (middle joint)
  19: PINKY_DIP  (joint near tip)
  20: PINKY_TIP  (fingertip)
WRIST: 0
'''


def draw_landmarks_on_image(image, hand_landmarks):
    h, w, _ = image.shape

    landmark_color = (0, 255, 0)  # Green
    connection_color = (255, 255, 255)  # White
    landmark_radius = 4
    connection_thickness = 2

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17), (17, 0)
    ]

    points = []
    h, w, _ = image.shape
    for landmark in hand_landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        points.append((x, y))

    for start, end in connections:
        cv2.line(
            image, points[start], points[end], connection_color, connection_thickness
        )
    for i, point in enumerate(points):
        if i in [4, 8, 12, 16, 20]:
            finger_tips_color = (255, 0, 0) #red
            cv2.circle(image, point, landmark_radius+1, finger_tips_color, connection_thickness)
        elif i == 0:
            wrist_color = (0, 0, 255) #blue
            cv2.circle(image, point, landmark_radius+2, wrist_color, connection_thickness)
        else:
            cv2.circle(image, point, landmark_radius, landmark_color, connection_thickness)

    return image


def draw_hand_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    detection_result = detector.detect(mp_image)

    for hand_landmarks in detection_result.hand_landmarks:
        frame = draw_landmarks_on_image(frame, hand_landmarks)
    return frame