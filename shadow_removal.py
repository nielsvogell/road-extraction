import cv2
import numpy as np
from matplotlib import pyplot as plt

# https://medium.com/arnekt-ai/shadow-removal-with-open-cv-71e030eadaf5

def main():

    name = "paper_exp.png"
    kernel_size = 105
    blur_size = 31

    # GET RGB IMAGE
    img = cv2.imread("data/" + name)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img_rgb)
    plt.title("img_rgb")
    plt.show()

    # SPLIT INTO COLOR PLANES
    rgb_planes = cv2.split(img_rgb)
    result_norm_planes = []

    for plane in rgb_planes:
        plt.imshow(plane)
        plt.title("plane")
        # plt.show()
        # DILATION
        img_dilation = cv2.dilate(plane, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
        plt.imshow(img_dilation)
        plt.title("img_dilation")
        # plt.show()
        # MEDIAN BLUR
        img_medianblur = cv2.medianBlur(img_dilation, blur_size)
        plt.imshow(img_medianblur)
        plt.title("img_medianblur")
        # plt.show()
        # ABSOLUTE DIFFERENCE
        img_diff = 255 - cv2.absdiff(plane, img_medianblur)
        plt.imshow(img_diff)
        plt.title("img_diff")
        # plt.show()
        # NORMALIZED IMAGE
        img_norm = cv2.normalize(img_diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        plt.imshow(img_norm)
        plt.title("img_norm")
        # plt.show()
        # ADD TO RESULT
        result_norm_planes.append(img_norm)

    img_removed_shadow = cv2.merge(result_norm_planes)
    plt.imshow(img_removed_shadow)
    plt.title("img_removed_shadow")
    plt.show()


if __name__ == "__main__":
    main()
