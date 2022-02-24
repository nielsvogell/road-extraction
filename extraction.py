import numpy as np
import cv2
import segmentation
import pickle
from matplotlib import use
use('TkAgg')
import matplotlib.pyplot as plt


img_rgb = cv2.cvtColor(cv2.imread('data/exp.png'), cv2.COLOR_BGR2RGB)
label_img, final_label = segmentation.segment(img_rgb)

# segmentation_obj = (label_img, final_label)
# label_file = open('segmented.pck', 'ab')
# pickle.dump(segmentation_obj, label_file)
# label_file.close()
# label_file = open('segmented.pck', 'rb')
# segmentation_obj = pickle.load(label_file)
# (label_img, final_label) = segmentation_obj
# label_file.close()


def keep_second_largest_component(img_lbl):
    no_components, components = cv2.connectedComponents(img_lbl)
    component_size = np.array([np.sum(components == lbl) for lbl in range(no_components)])
    second_largest_component = np.argpartition(component_size.flatten(), -2)[-2]
    mask = np.zeros_like(components, dtype=np.uint8)
    mask[components == second_largest_component] = 1
    return mask


plt.imshow(img_rgb)


road_mask = np.zeros_like(label_img).astype(np.uint8)
road_mask[np.where(label_img == final_label['road'])] = 255

plt.imshow(road_mask, cmap='gray')

# Find connected components
no_components, components = cv2.connectedComponents(road_mask)
plt.clf()
plt.imshow(components)

# remove those with small surface area
component_size = np.array([np.sum(components == lbl) for lbl in range(no_components)])
road_component = np.argpartition(component_size.flatten(), -2)[-2]
road_mask_main = np.zeros_like(components, dtype=np.uint8)
road_mask_main[components == road_component] = 1

plt.clf()
plt.imshow(road_mask_main)

# use morph close to close holes and remove noise
roads_closed = cv2.morphologyEx(road_mask_main, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

plt.clf()
plt.imshow(roads_closed)

# use open to cut away pathways with a structuring element that is smaller than the main road (main road width ~20px)
roads_opened = cv2.morphologyEx(road_mask_main, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
plt.clf()
plt.imshow(roads_opened)
roads_opened = keep_second_largest_component(roads_opened)
plt.clf()
plt.imshow(roads_opened)
roads_opened = cv2.morphologyEx(roads_opened, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)))
plt.clf()
plt.imshow(roads_opened)

# Thinning process
roads_inv = np.ones_like(roads_opened, dtype=np.uint8)
roads_inv[roads_opened == 1] = 0
roads_inv = cv2.dilate(roads_inv, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
border = np.logical_and(roads_inv, roads_opened)
plt.clf()
plt.imshow(border, cmap='gray')

ker_nb = np.ones((3, 3), dtype=np.uint8)
ker_nb[1, 1] = 0
border_filtered = cv2.filter2D(border.astype(np.uint8), 0, kernel=ker_nb)
border_remove = border.copy()
border_remove[np.where(border_filtered != 2)] = 0
plt.imshow(border, cmap='gray')
roads_thinned = roads_opened.copy()
roads_thinned[np.where(border_remove == 1)] = 0
plt.imshow(roads_opened, cmap='gray')
plt.imshow(roads_thinned, cmap='gray')


def skeletonize(mask):
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


# Find intersection points
roads_thinned = skeletonize(roads_opened)

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

