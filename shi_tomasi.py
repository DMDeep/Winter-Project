import cv2
import numpy as np

img = cv2.imread('trial3.jpg')
img = cv2.resize(img, (500, 500))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray, 10, 0.01, 10)
corners = np.int0(corners)

for one_corner in corners:
    x, y = one_corner.ravel()
    cv2.circle(img, (x, y), 6, (255, 0, 0), -1)

cv2.imshow('result', img)    

cv2.waitKey(0)
cv2.destroyAllWindows()