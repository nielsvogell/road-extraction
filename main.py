import numpy as np
import cv2
from segmentation import segment
from building_extraction import extract_buildings
from road_extraction import extract_roads
import sys
if 'matplotlib' not in sys.modules:
    from matplotlib import use
    use('TkAgg')
from matplotlib import pyplot as plt


def main():
    img_rgb = cv2.imread('data/munich.png')

    labeled_img, label_colors = segment(img_rgb)
    roads = extract_roads(labeled_img)
    buildings = extract_buildings(labeled_img, output_type='mask')
    
    img_out = img_rgb.copy()
    img_out[np.where(buildings > 0)] = np.array([250, 120, 120], dtype=np.uint8)
    img_out[np.where(roads > 0)] = np.array([250, 250, 250], dtype=np.uint8)
    
    plt.imshow(img_out)
    
    
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
    print('Done')

