# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def main():
    img1 = cv2.imread('data/sat15.png')
    img_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    plt.imshow(img_gray, cmap='gray')
    _, thresh = cv2.threshold(img_gray, 100, 255, type=cv2.THRESH_BINARY)
    plt.imshow(thresh, cmap='gray')
    thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    plt.imshow(thresh, cmap='gray')
    kernel = np.ones((5, 5), np.uint8)
    thresh_morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    thresh_morph = cv2.morphologyEx(thresh_morph, cv2.MORPH_CLOSE, kernel)
    plt.imshow(thresh_morph, cmap='gray')
    
    # remove green
    img_tmp = img1.copy()
    img_tmp[:, :, 1] = 0
    img_tmp_gray = cv2.cvtColor(img_tmp, cv2.COLOR_BGR2GRAY)
    thresh_tmp = cv2.adaptiveThreshold(img_tmp_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, 9, 2)
    kernel = np.ones((3, 13), np.uint8)
    thresh_tmp = cv2.morphologyEx(thresh_tmp, cv2.MORPH_OPEN, kernel, iterations=1)
    plt.imshow(thresh_tmp, cmap='gray')
    kernel = np.ones((5, 1), np.uint8)
    thresh_tmp = cv2.morphologyEx(thresh_tmp, cv2.MORPH_OPEN, kernel, iterations=1)
    plt.imshow(thresh_tmp, cmap='gray')

    thresh_tmp = cv2.erode(thresh_tmp, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh_tmp = cv2.dilate(thresh_tmp, kernel)
    thresh_tmp = cv2.morphologyEx(thresh_tmp, cv2.MORPH_CLOSE, kernel, iterations=3)
    plt.imshow(thresh_tmp, cmap='gray')
    
    thresh_smooth = cv2.GaussianBlur(thresh_morph, (5, 5), 2)
    edge = cv2.Canny(img_gray, 25, 75, None, 3)
    plt.imshow(edge, cmap='gray')
    linesP = cv2.HoughLinesP(edge, 4, np.pi / 180, 60, None, 40, 15)
    
    img_bilateral = cv2.bilateralFilter(img_gray, 9, 75, 75)
    plt.imshow(img_bilateral, cmap='gray')

    img_out = cv2.cvtColor(thresh_morph.copy(), cv2.COLOR_GRAY2BGR)
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(img_out, (l[0], l[1]), (l[2], l[3]), (255, 255, 0), 3, cv2.LINE_AA)
    plt.imshow(img_out)
    
    
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    plt.ion()
    plt.draw()
    main()
    print('Done')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
