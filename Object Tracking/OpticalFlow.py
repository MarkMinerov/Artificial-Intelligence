import cv2
import numpy as np
from time import sleep

cap = cv2.VideoCapture(0)

ret, frame1 = cap.read()
prev_img = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
hsv_mask = np.zeros_like(frame1)
hsv_mask[:, :, 1] = 255

while True:
    ret, frame2 = cap.read()
    if not ret:
        break

    next_img = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prev_img, next_img, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(
        flow[:, :, 0], flow[:, :, 1], angleInDegrees=True)
    hsv_mask[:, :, 0] = ang / 2
    hsv_mask[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2RGB)

    cv2.imshow('Window', bgr)
    sleep(1 / 60)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    prev_img = next_img

cv2.destroyAllWindows()
cap.release()
