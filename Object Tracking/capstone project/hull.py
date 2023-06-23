import cv2
import numpy as np
from sklearn.metrics import pairwise

# Create a list of points
points = np.array([[100, 50], [200, 50], [300, 0], [150, 200], [50, 150], [250, 100]], dtype=np.int32)

# Compute the convex hull of the points
hull = cv2.convexHull(points)

# Draw the convex hull on a black image
image = np.zeros((300, 300, 3), dtype=np.uint8)

# the most extreme 4 points
top = tuple(hull[hull[:, :, 1].argmin()][0])
bottom = tuple(hull[hull[:, :, 1].argmax()][0])
left = tuple(hull[hull[:, :, 0].argmin()][0])
right = tuple(hull[hull[:, :, 0].argmax()][0])


cv2.polylines(image, [hull], isClosed=True, color=(0, 255, 0), thickness=2)

for point in points:
    cv2.circle(image, point, 4, (255, 0, 255), -1)

cv2.circle(image, top, 4, (255, 0, 0), -1)
cv2.circle(image, bottom, 4, (255, 0, 0), -1)
cv2.circle(image, left, 4, (255, 0, 0), -1)
cv2.circle(image, right, 4, (255, 0, 0), -1)

cX = (left[0] + right[0]) // 2
cY = (top[1] + bottom[1]) // 2

cv2.circle(image, (cX, cY), 4, (127, 255, 255), -1)

distance = pairwise.euclidean_distances([(cX, cY)], [left, right, top, bottom])[0]
max_distance = distance.max()
radius = int(0.9*max_distance)
circumference = 2 * np.pi * radius
cv2.circle(image, (cX, cY), radius, 255, 10)


# Display the image with the convex hull
cv2.imshow("Convex Hull", image)
cv2.waitKey(0)
cv2.destroyAllWindows()