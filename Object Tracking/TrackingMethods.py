import cv2

trackers = [attr for attr in dir(cv2) if attr.startswith(
    'Tracker') and attr.endswith('create')]

NOP_FRAMES_LIMIT = 30


def select_method():
    print("Choose Tracker:")

    for (i, tracker) in enumerate(trackers):
        print(f"For tracker {str(tracker)} enter {i}")

    index = int(input("Choose index: "))

    return trackers[index]


tracker_name = select_method()
tracker = eval(f"cv2.{tracker_name}()")

cap = cv2.VideoCapture(0)
nop_frames = 0

while nop_frames < NOP_FRAMES_LIMIT:
    ret, frame = cap.read()
    nop_frames += 1

ret, frame = cap.read()
# roi = cv2.selectROI(frame, False)
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

face_rects = face_cascade.detectMultiScale(frame)

if len(face_rects) == 0:
    exit()

(face_x, face_y, w, h) = tuple(face_rects[0])
track_window = (face_x, face_y, w, h)

ret = tracker.init(frame, track_window)

while True:
    ret, frame = cap.read()
    success, roi = tracker.update(frame)
    (x, y, w, h) = tuple(map(int, roi))

    if success:
        p1 = (x, y)
        p2 = (x + w, y + h)
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 5)
    else:
        cv2.putText(frame, "Failure to Detect ROI!", (100, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    cv2.imshow(tracker_name, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
