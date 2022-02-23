import numpy as np
import cv2
import segmentation
from matplotlib import use
use('TkAgg')
import matplotlib.pyplot as plt

img_rgb = cv2.cvtColor(cv2.imread('data/exp.png'), cv2.COLOR_BGR2RGB)
label_img, final_label = segmentation.segment(img_rgb)

road_mask = np.zeros_like(label_img)

plt.imshow(label_img)

