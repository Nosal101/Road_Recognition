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
    lower_green2 = np.array([0, 105, 0], dtype=np.uint8)
    upper_green2 = np.array([72, 255, 237], dtype=np.uint8)
    lower_blue = np.array([0, 0, 255], dtype=np.uint8)
    upper_blue = np.array([133, 61, 255], dtype=np.uint8)
    lower_blue2 = np.array([0, 0, 255], dtype=np.uint8)
    upper_blue2 = np.array([91, 87, 255], dtype=np.uint8)
    lower_blue3 = np.array([88, 65, 198], dtype=np.uint8)
    upper_blue3 = np.array([146, 173, 255], dtype=np.uint8)
    # Stwórz maski
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_blue2 = cv2.inRange(hsv, lower_blue2, upper_blue2)
    mask_green2 = cv2.inRange(hsv, lower_green2, upper_green2)
    mask_blue3 = cv2.inRange(hsv, lower_blue3, upper_blue3)
    # Połącz 2 maski ze sobą
    combined_mask = cv2.bitwise_or(mask_green, mask_blue)
    combined_mask = cv2.bitwise_or(combined_mask, mask_blue2)
    combined_mask = cv2.bitwise_or(combined_mask, mask_green2)
    combined_mask = cv2.bitwise_or(combined_mask, mask_blue3)
    #Wyciecie narożników
    kolor_kwadratu = (255, 255, 255)
    kwadrat = np.ones((200, 300, 3), dtype=np.uint8)
    kwadrat[:] = kolor_kwadratu
    img[0:200, 0:300] = kwadrat
    img[0:200, -300:] = kwadrat
    # Nałóż maski na obraz i przeprowadz progowanie
    img_with_mask = cv2.bitwise_and(img, img, mask=combined_mask)
    _, thresh = cv2.threshold(img_with_mask, 0, 255, cv2.THRESH_BINARY)
    thresh_inverted = cv2.bitwise_not(thresh)
    # Wygładz obrazy by nie było czarnych ani białych kropek
    kernel = np.ones((10, 10), np.uint8)
    opening = cv2.morphologyEx(thresh_inverted, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)
    closing_inverted = cv2.bitwise_not(closing)
    # Odwróć kolory i nałóż obrazy na siebie
    result = cv2.bitwise_or(img, closing_inverted)
    # Przedstaw obraz w skali szarości
    gray = cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
    gray_inverted = cv2.bitwise_not(gray)
    #Treshold
    tresh_value = 220
    mask = gray_inverted > tresh_value
    tresh_value2 = 40
    mask2 = gray_inverted < tresh_value2
    gray_inverted[mask] = 0
    gray_inverted[mask2] = 0
    # Wyświetl obraz
    plt.imshow(gray_inverted,cmap='gray')
    plt.show()



index=0
while True:
    display_image(index)
    index+=5