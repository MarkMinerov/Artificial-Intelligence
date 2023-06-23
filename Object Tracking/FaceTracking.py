import numpy as np
import cv2
from time import sleep

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(1)
sleep(1)
ret, frame = cap.read()
face_rects = face_cascade.detectMultiScale(frame)
(face_x, face_y, w, h) = tuple(face_rects[0])
track_window = (face_x, face_y, w, h)

roi = frame[face_y:face_y + h, face_x:face_x + w]

hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
# image
# channel (hue as [0])
# mask=None
# [180] number of bins
# [0, 180] range of possible hue values
roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# cv2.TERM_CRITERIA_EPS: This flag indicates that the termination criteria is based on the desired accuracy (epsilon).
# cv2.TERM_CRITERIA_COUNT: This flag indicates that the termination criteria is based on the maximum number of iterations.
# 10: This parameter specifies the maximum number of iterations allowed. In this case, it is set to 10, meaning the iterative algorithm will terminate after 10 iterations if the other termination condition is not met.
# 1: This parameter specifies the desired accuracy (epsilon) for the iterative algorithm. It is used when cv2.TERM_CRITERIA_EPS flag is set. In this case, it is set to 1, indicating a relatively low level of accuracy.
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # [hsv] - The input image in the HSV color space
    # [0] - hue channel of the image
    # roi_hist - The histogram of the region of interest (ROI) previously calculated from the face
    # [0, 180] - The range of values for the histogram bins. In the HSV color space, the hue channel values range from 0 to 180
    # 1 - The scaling factor applied to the back projection histogram (no scale)

    # Back projection is a technique that highlights regions in the image that have similar pixel
    # distributions to the histogram. The resulting back projection image is stored in the variable dst.
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
    # dst - The back projection image calculated in the previous step.
    # track_window - The current position and size of the tracked object's bounding box.
    # It represents the region to search for the object in the current frame.
    # term_crit - The termination criteria for the mean-shift algorithm, as defined earlier.

    # The cv2.meanShift function performs the mean-shift algorithm on the back projection
    # image dst to track the object within the specified track_window. It searches for the
    # peak in the back projection image and updates the position of the track_window to the
    # new location of the object. The function returns two values: ret, which indicates whether
    # the algorithm successfully converged, and track_window, which contains the updated position
    # and size of the tracked object's bounding box.

    ret, track_window = cv2.meanShift(dst, track_window, term_crit)
    x, y, w, h = track_window
    img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)

    # ret, track_window = cv2.CamShift(dst, track_window, term_crit)
    # pts = cv2.boxPoints(ret)
    # pts = np.int0(pts)
    # img2 = cv2.polylines(frame, [pts], True, (255, 127, 0), 5)

    cv2.imshow('Image', img2)
    sleep(1 / 60)  # fps

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
