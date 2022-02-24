import numpy as np
import cv2
from matplotlib import use
use('TkAgg')
import matplotlib.pyplot as plt
import segmentation
import pickle


def main():
    img_rgb = cv2.cvtColor(cv2.imread('data/munich.png'), cv2.COLOR_BGR2RGB)
    label_img, final_label = segmentation.segment(img_rgb, nr_clusters=5)

    plt.imshow(img_rgb)
    plt.imshow(label_img)
    # segmentation_obj = (label_img, final_label)
    # label_file = open('segmented.pck', 'ab')
    # pickle.dump(segmentation_obj, label_file)
    # label_file.close()
    # label_file = open('segmented.pck', 'rb')
    # segmentation_obj = pickle.load(label_file)
    # (label_img, final_label) = segmentation_obj
    # label_file.close()
    
    road_mask = np.zeros_like(label_img).astype(np.uint8)
    road_mask[np.where(label_img == final_label['road'])] = 255

    road_mask_final = process_road_mask(road_mask)

    # Thinning process
    roads_thinned = get_skeleton(road_mask_final)

    # Find intersection points
    plt.imshow(roads_thinned, cmap='gray')
    # Scan for points with at least 3 neighbors
    ker_eight_nb = np.ones((3, 3), dtype=np.uint8)
    ker_eight_nb[1, 1] = 0
    ker_four_nb = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    neighbors_eight = cv2.filter2D(roads_thinned, 0, kernel=ker_eight_nb)
    neighbors_four = cv2.filter2D(roads_thinned, 0, kernel=ker_four_nb)
    c = neighbors_four.astype(np.int32)
    ker_top_right = cv2.filter2D(roads_thinned, 0, kernel=np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]]))
    ker_top_right_cor = cv2.erode(roads_thinned, kernel=np.array([[0, 0, 1], [0, 1, 0], [0, 0, 0]], dtype=np.uint8))
    plt.imshow(ker_top_right_cor, cmap='gray')
    c += np.where(np.logical_and(ker_top_right_cor, ker_top_right), 1, 0).astype(np.int32)
    ker_bottom_right = cv2.filter2D(roads_thinned, 0, kernel=np.array([[0, 0, 0], [0, 1, 0], [0, 1, 0]])) == 0
    ker_bottom_right_cor = cv2.erode(roads_thinned, kernel=np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8))
    c += np.where(np.logical_and(ker_bottom_right_cor, ker_bottom_right), 1, 0).astype(np.int32)
    ker_bottom_left = cv2.filter2D(roads_thinned, 0, kernel=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])) == 0
    ker_bottom_left_cor = cv2.erode(roads_thinned, kernel=np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=np.uint8))
    c += np.where(np.logical_and(ker_bottom_right_cor, ker_bottom_right), 1, 0).astype(np.int32)
    ker_top_left = cv2.filter2D(roads_thinned, 0, kernel=np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])) == 0
    ker_top_left_cor = cv2.erode(roads_thinned, kernel=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.uint8))
    c += np.where(np.logical_and(ker_bottom_right_cor, ker_bottom_right), 1, 0).astype(np.int32)

    plt.clf()
    plt.imshow(c > 3, cmap='gray')


def keep_n_largest_components(img_labeled, n_largest=1):
    # Find connected components
    no_components, components = cv2.connectedComponents(img_labeled)
    # Calculate surface area
    component_size = np.array([np.sum(components == lbl) for lbl in range(no_components)])
    # take the n_largest largest components to continue
    if n_largest > no_components:
        n_largest = no_components
    n_largest_components = np.argpartition(component_size.flatten(), -(n_largest + 1))[-(n_largest + 1):]
    n_largest_components = np.sort(n_largest_components)[:-1]
    
    # Create mask for remaining components
    mask = np.zeros_like(img_labeled, dtype=np.uint8)
    for lbl in n_largest_components:
        mask[np.where(np.logical_or(mask, components == lbl))] = 1
        
    return mask


def process_road_mask(img_labeled, n_largest=1):
    # Get the largest connected components
    roads_processed = keep_n_largest_components(img_labeled, n_largest=n_largest)
    
    # use morph close to close holes and remove noise
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    roads_processed = cv2.morphologyEx(roads_processed, cv2.MORPH_CLOSE, kernel=ker)
    
    # use open to cut away pathways with a structuring element that is smaller than the main road
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    roads_processed = cv2.morphologyEx(roads_processed, cv2.MORPH_OPEN, kernel=ker)
    
    # After cutting into the roads, remove possibly new, smaller components
    roads_processed = keep_n_largest_components(roads_processed, n_largest)
    
    # Close any potential new holes in the mask
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    roads_processed = cv2.morphologyEx(roads_processed, cv2.MORPH_CLOSE, kernel=ker)
    
    return roads_processed


def get_skeleton(mask):
    # 111
    # 101
    # 111
    ker_nb = np.ones((3, 3), dtype=np.uint8)
    ker_nb[1, 1] = 0
    
    removed_points = 1
    while removed_points > 0:
        mask_inv = np.ones_like(mask, dtype=np.uint8)
        mask_inv[np.where(mask == 1)] = 0
        mask_inv = cv2.dilate(mask_inv, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=1)
        
        mask_filtered = cv2.filter2D(mask, 0, kernel=ker_nb)
        
        border = np.logical_and(mask_inv, mask)
        border_filtered = cv2.filter2D(border.astype(np.uint8), 0, kernel=ker_nb)
        border_remove = border.copy()
        border_remove[np.where(np.logical_or(border_filtered != 2, mask_filtered <= border_filtered))] = 0
        mask_thinned = mask.copy()
        mask_thinned[np.where(border_remove == 1)] = 0
        # plt.clf()
        # plt.imshow(mask_thinned, cmap='gray')
        removed_points = np.sum(np.logical_and(mask, mask_thinned == 0))
        mask = mask_thinned.copy()
        # plt.clf()
        # plt.imshow(mask, cmap='gray')
    
    # TODO: The diagonals could be one pixel thinner
    return mask


if __name__ == '__main__':
    main()

