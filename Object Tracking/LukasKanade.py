import cv2
import numpy as np
from time import sleep

corner_track_params = {
    "maxCorners": 5,
    "qualityLevel": 0.3,
    "minDistance": 7,
    "blockSize": 7
}

lk_params = {
    "winSize": (200, 200),
    "maxLevel": 2,
    "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
}

cap = cv2.VideoCapture(0)
sleep(1)
ret, prev_frame = cap.read()

saved_index = 0


def draw_good_feature(event, x, y, flags, param):
    global saved_index
    global prev_frame

    if event == cv2.EVENT_FLAG_LBUTTON:
        cv2.circle(prev_frame, (x, y), 8, (255, 0, 0), -1)
        prev_pts[saved_index] = [[x, y]]
        saved_index += 1


# prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **corner_track_params)
prev_pts = np.zeros(
    (corner_track_params["maxCorners"], 1, 2)).astype(np.float32)

cv2.namedWindow('Window')
cv2.setMouseCallback('Window', draw_good_feature)

while True:
    cv2.imshow('Window', prev_frame)

    if cv2.waitKey(1) & 0xFF == ord('c'):
        break

if saved_index == 0:
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
mask = np.zeros_like(prev_frame)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    next_pts, status, err = cv2.calcOpticalFlowPyrLK(
        prev_gray, frame_gray, prev_pts, None, **lk_params)

    good_new = next_pts[status == 1]
    good_prev = prev_pts[status == 1]

    for i, (new, prev) in enumerate(zip(good_new, good_prev)):
        x_new, y_new = map(int, new.ravel())
        x_prev, y_prev = map(int, prev.ravel())

        cv2.line(mask, (x_new, y_new),
                 (x_prev, y_prev), (127, 0, 255), 3)
        frame = cv2.circle(frame, (x_new, y_new), 8, (0, 255, 0), -1)

    img = cv2.add(frame, mask)
    cv2.imshow('Window', img)
    sleep(1 / 60)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    prev_gray = frame_gray.copy()
    prev_pts = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()
cap.release()
