import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Utwórz listę plików w folderze
files = os.listdir("image_2")
files.sort()
# Funkcja do wyświetlania obrazów
def display_image(index):
    # Wczytaj zdjęcie z pliku
    file = files[index]
    img = cv2.imread(os.path.join("image_2", file))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Ustaw granice kolorów
    lower_green = np.array([40, 40, 40], dtype=np.uint8)
    upper_green = np.array([70, 255, 255], dtype=np.uint8)
    lower_blue = np.array([0, 0, 255], dtype=np.uint8)
    upper_blue = np.array([133, 61, 255], dtype=np.uint8)
    # Stwórz maski
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    combined_mask = cv2.bitwise_or(mask_green, mask_blue)
    # Nałóż maski na obraz i przeprowadz progowanie
    img_with_mask = cv2.bitwise_and(img, img, mask=combined_mask)
    _, thresh = cv2.threshold(img_with_mask, 0, 255, cv2.THRESH_BINARY)
    thresh_inverted = cv2.bitwise_not(thresh)
    # Wygładz obrazy by nie było czarnych ani białych kropek
    kernel = np.ones((10, 10), np.uint8)
    opening = cv2.morphologyEx(thresh_inverted, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    closing_inverted = cv2.bitwise_not(closing)
    # Odwróć kolory i nałóż obrazy na siebie
    result = cv2.bitwise_or(img, closing_inverted)

    # Wyświetl obraz
    plt.imshow(result)
    plt.show()
index=0
while True:
    display_image(index)
    index+=1