import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

img = cv2.imread("image_2/um_000001.png")

# Wyświetl obraz
#plt.imshow(img)
#plt.show()

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_green = np.array([40, 40, 40], dtype=np.uint8)
upper_green= np.array([70, 255, 255], dtype=np.uint8)

mask = cv2.inRange(hsv, lower_green, upper_green)

img_without_green = cv2.bitwise_and(img, img, mask = mask)
_, thresh = cv2.threshold(img_without_green, 0, 255, cv2.THRESH_BINARY)
thresh_inverted = cv2.bitwise_not(thresh)

# Wyświetl obraz
#plt.imshow(thresh_inverted)
#plt.show()

kernel = np.ones((6,6),np.uint8)
opening = cv2.morphologyEx(thresh_inverted, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
closing_inverted = cv2.bitwise_not(closing)

# Wyświetl obraz
#plt.imshow(closing_inverted)
#plt.show()

result = cv2.bitwise_or(img, closing_inverted)

# Wyświetl obraz
plt.imshow(result)
plt.show()