import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def nothing(x):
    pass

# Utwórz okno
cv2.namedWindow('image')

# Stwórz suwaki dla dolnego i górnego zakresu kolorów
cv2.createTrackbar('Low H', 'image', 0, 255, nothing)
cv2.createTrackbar('High H', 'image', 0, 255, nothing)
cv2.createTrackbar('Low S', 'image', 0, 255, nothing)
cv2.createTrackbar('High S', 'image', 0, 255, nothing)
cv2.createTrackbar('Low V', 'image', 0, 255, nothing)
cv2.createTrackbar('High V', 'image', 0, 255, nothing)

# Wczytaj obraz
image = cv2.imread("image_2/um_000018.png")

while(1):
    # Konwertuj BGR do HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Odczytaj wartości suwaków
    low_h = cv2.getTrackbarPos('Low H', 'image')
    high_h = cv2.getTrackbarPos('High H', 'image')
    low_s = cv2.getTrackbarPos('Low S', 'image')
    high_s = cv2.getTrackbarPos('High S', 'image')
    low_v = cv2.getTrackbarPos('Low V', 'image')
    high_v = cv2.getTrackbarPos('High V', 'image')

    # Zdefiniuj dolny i górny zakres kolorów
    lower = np.array([low_h, low_s, low_v], dtype=np.uint8)
    upper= np.array([high_h, high_s, high_v], dtype=np.uint8)

    # Stwórz maskę kolorów
    mask = cv2.inRange(hsv, lower, upper)

    # Nałóż maskę na oryginalny obraz
    res = cv2.bitwise_and(image, image, mask=mask)

    # Wyświetl obraz
    cv2.imshow('image', res)

    # Czekaj na naciśnięcie klawisza 'Esc' (kod ASCII dla Esc: 27)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()