import cv2, math
import numpy as np
import mediapipe as mp

cv2.namedWindow('PushUp Project')
camera = cv2.VideoCapture(0)

CAMERA_WIDTH = 1280         # <- CHANGE HERE TO 1920 X 1080
CAMERA_HEIGHT = 720

camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

VALID_ANGLE_DOWN = 90
VALID_ANGLE_UP = 145

CIRCLE_LINES_COLORS = (0, 85, 255)
TEXT_COLOR = (0, 0, 0)
ANGLE_COLOR = (0, 0, 0)

number_of_pushups = 0
epsilon = 1e-6
state = "up"    # state just keep the current state of the pushup : 'up' / 'down'
go = "down"     # go will let you know when it's time to get down or up on time/rithm : 'up' / 'down'

use_valid = True        # if i want to use a {valid_frames} frame validation per pushup
valid_frames = 1        # to validate a push-up    |||   to make them more difficult :  from 2 -> 5 - 10 even more
ld_cnt, rd_cnt = 0, 0   # left/right arm down  counter
lu_cnt, ru_cnt = 0, 0   # left/right arm up    counter

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)


# we calculate the angle at elbow using : Cosine Theorem and Distance Between Points
def angle_at_elbow(landmarks, a_idx, b_idx, c_idx):

    # 3D coord. for better accuracy  |  we use NumPy arrays for easier math
    A = np.array([landmarks[a_idx].x, landmarks[a_idx].y, landmarks[a_idx].z], dtype=float)
    B = np.array([landmarks[b_idx].x, landmarks[b_idx].y, landmarks[b_idx].z], dtype=float)
    C = np.array([landmarks[c_idx].x, landmarks[c_idx].y, landmarks[c_idx].z], dtype=float)

    # helper function for distance between 3 points in space
    def distance(p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

    # length of each side of the triangle
    a = distance(C, B)
    b = distance(A, C)
    c = distance(A, B)

    if a < epsilon or c < epsilon:
        return None

    # Law of cosine
    cos = (a**2 + c**2 - b**2) / (2*a*c)
    cos = np.clip(cos, -1.0, 1.0)  # we get cos to be in [-1.0, 1.0] range
    angle = math.degrees(math.acos(cos))

    return angle


def existing_points(landmarks, ids, visibility_threshold=0.5):
    for i in ids:
        if landmarks[i].visibility < visibility_threshold:
            return False
    return True


while True:
    ok, frame = camera.read() if camera.isOpened() else (False, None)
    if not ok:
        break

    frame = cv2.flip(frame, 1)  # we invert the camera

    frame_copy = frame.copy()
    result = pose.process(frame_copy)

    if result.pose_landmarks:
        landmark = result.pose_landmarks.landmark

        # mediapipe constant presets
        LEFT_SHOULD, RIGHT_SHOULD = 11, 12
        LEFT_ELBOW, RIGHT_ELBOW = 13, 14
        LEFT_WRIST, RIGHT_WRIST = 15, 16

        # we check if the points are present in the current frame:
        if existing_points(landmark, [LEFT_SHOULD, RIGHT_SHOULD, LEFT_WRIST, RIGHT_WRIST, LEFT_ELBOW, RIGHT_ELBOW]):

            # and if we calculate their angles
            angle_left_arm = angle_at_elbow(landmark, LEFT_SHOULD, LEFT_ELBOW, LEFT_WRIST)
            angle_right_arm = angle_at_elbow(landmark, RIGHT_SHOULD, RIGHT_ELBOW, RIGHT_WRIST)

            # draw the points
            for id in [LEFT_SHOULD, LEFT_ELBOW, LEFT_WRIST, RIGHT_WRIST, RIGHT_ELBOW, RIGHT_SHOULD]:
                x, y = int(landmark[id].x * CAMERA_WIDTH), int(landmark[id].y * CAMERA_HEIGHT)
                cv2.circle(frame_copy, (x, y), 12, CIRCLE_LINES_COLORS, -1)

            # draw lines connecting the points
            for stt, end in [[LEFT_SHOULD, LEFT_ELBOW], [LEFT_ELBOW, LEFT_WRIST], [RIGHT_SHOULD, RIGHT_ELBOW], [RIGHT_ELBOW, RIGHT_WRIST]]:
                x1, y1 = int(landmark[stt].x * CAMERA_WIDTH), int(landmark[stt].y * CAMERA_HEIGHT)
                x2, y2 = int(landmark[end].x * CAMERA_WIDTH), int(landmark[end].y * CAMERA_HEIGHT)

                cv2.line(frame_copy, (x1, y1), (x2, y2), CIRCLE_LINES_COLORS, 3)

            # draw the angle at elbow
            if angle_left_arm is not None:
                x, y = int(landmark[LEFT_ELBOW].x * CAMERA_WIDTH), int(landmark[LEFT_ELBOW].y * CAMERA_HEIGHT)
                cv2.putText(frame_copy, f"{int(angle_left_arm)}", (x-30, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, ANGLE_COLOR, 4, cv2.LINE_AA)

            if angle_right_arm is not None:
                x, y = int(landmark[RIGHT_ELBOW].x * CAMERA_WIDTH), int(landmark[RIGHT_ELBOW].y * CAMERA_HEIGHT)
                cv2.putText(frame_copy, f"{int(angle_right_arm)}", (x-30, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, ANGLE_COLOR, 4, cv2.LINE_AA)

            # we count the push-ups
            if angle_right_arm is not None and angle_left_arm is not None:
                if use_valid:  # to make the pushup harder
                    if angle_left_arm < VALID_ANGLE_DOWN: ld_cnt += 1
                    else: ld_cnt = 0

                    if angle_right_arm < VALID_ANGLE_DOWN: rd_cnt += 1
                    else: rd_cnt = 0

                    if angle_left_arm > VALID_ANGLE_UP: lu_cnt += 1
                    else: lu_cnt = 0

                    if angle_right_arm > VALID_ANGLE_UP: ru_cnt += 1
                    else: ru_cnt = 0

                    both_arms_down = (ld_cnt >= valid_frames and rd_cnt >= valid_frames)
                    both_arms_up = (lu_cnt >= valid_frames and ru_cnt >= valid_frames)
                else:
                    both_arms_down = (angle_left_arm < VALID_ANGLE_DOWN and angle_right_arm < VALID_ANGLE_DOWN)
                    both_arms_up = (angle_left_arm > VALID_ANGLE_UP and angle_right_arm > VALID_ANGLE_UP)

                if state == "up" and both_arms_down:
                    state = "down"
                    go = "up"
                elif state == "down" and both_arms_up:
                    state = "up"
                    go = "down"
                    number_of_pushups += 1

                # clap once to reset the counter for the next person
                if 0.02 < abs(landmark[RIGHT_WRIST].x - landmark[LEFT_WRIST].x) < 0.1 and 0.02 < abs(landmark[RIGHT_WRIST].y - landmark[LEFT_WRIST].y) < 0.1:
                    number_of_pushups = 0  # break

    cv2.putText(frame_copy, f"Pushups: {number_of_pushups}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, TEXT_COLOR, 3, cv2.LINE_AA)
    cv2.putText(frame_copy, f"State: {go}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 3, cv2.LINE_AA)

    cv2.imshow('PushUp Project', frame_copy)
    if cv2.waitKey(1) == ord('q'):
        break


camera.release()
cv2.destroyAllWindows()
