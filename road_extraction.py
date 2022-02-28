import numpy as np
import cv2
# import pickle  # for alternative approach to load GaussianMixtureModel from file, rather than recalculate it
# from importlib import reload  # for debuggin purposes
import sys
if 'matplotlib' not in sys.modules:
    from matplotlib import use
    use('TkAgg')  # The standard QtAgg causes problems on some machines
import segmentation
from matplotlib import pyplot as plt

# for debugging
# if False:
#     if 'extraction' in sys.modules:
#         reload(sys.modules['extraction'])
#     from road_extraction import keep_n_largest_components
#     from road_extraction import get_skeleton
#     from road_extraction import thinning_wang
#     from road_extraction import process_road_mask


def main():
    # Load example image from Jin et al. (2012) for comparison
    img_rgb = cv2.cvtColor(cv2.imread('data/example_lit.png'), cv2.COLOR_BGR2RGB)
    
    # Get the segmented labels from the input image
    cluster_img, cluster_labels, _ = segmentation.segment(img_rgb, nr_clusters=5)
    
    # Extract road network
    roads_thinned = extract_roads(cluster_img, cluster_labels)
    
    # Draw road network on image
    img_rgb[np.where(roads_thinned > 0)] = np.array([50, 50, 250], dtype=np.uint8)
    
    # Output and save figure
    plt.imshow(img_rgb, cmap='gray')
    plt.axis('off')
    plt.savefig('out/example_lit_out.png')
    
    
def extract_roads(cluster_img, cluster_labels, n_largest=1):
    """
    Extracts the road network from a segmented image
    :param cluster_img: 2d array with labels of each cluster at each pixel
    :param cluster_labels: dict containing the labels belonging roads, buildings and background
    :param n_largest: number of largest connected components considered for the road network
    :return: 2d array with ones where the thinned road network is and zeros else
    """
    # Create a mask of all clusters labeled 'road'
    road_mask = np.zeros_like(cluster_img).astype(np.uint8)
    for lbl in cluster_labels['road']:
        road_mask[np.where(cluster_img == lbl)] = 255
    
    # Remove small components, fill holes in the mask
    road_mask_final = process_road_mask(road_mask, n_largest=n_largest)
    # plt.imshow(road_mask_final, cmap='gray')
    
    # Thinning process
    # Get a skeleton representation of the road network
    road_network = get_skeleton(road_mask_final)
    # plt.imshow(road_network, cmap='gray')
    
    # Improve the thinning with an application of the wang89 algorithm
    # The algorithm is not yet fully correctly implemented due to ambiguities in the paper
    # roads_thinned_wang = thinning_wang(road_network)
    # plt.imshow(roads_thinned_wang, cmap='gray')
    
    # Find intersection points
    # Does currently not work properly because the skeleton is not thin enough
    # intersection_points = get_intersection_points(road_network)
    
    return road_network


def keep_n_largest_components(img_labeled, n_largest=1):
    # Find connected components
    no_components, components = cv2.connectedComponents(img_labeled.astype(np.uint8))
    # Calculate surface area
    component_size = np.array([np.sum(components == lbl) for lbl in range(no_components)])
    # take the n_largest largest components to continue
    if n_largest > no_components:
        n_largest = no_components
    if no_components == 1:  # prevents an error in the slicing that occurs if only one component is detected
        n_largest_components = [0]
    else:
        # np.argpartition splits the components in such a way that the n largest components occur at the end (unsorted)
        n_largest_components = np.argpartition(component_size.flatten(), -(n_largest + 1))[-(n_largest + 1):]
        # Assuming that the background (non-road) component fills the larges area, remove the largest component
        n_largest_components = np.delete(n_largest_components, np.argmax(component_size[n_largest_components]))
    
    # Create mask for remaining components
    mask = np.zeros_like(img_labeled, dtype=np.uint8)
    for lbl in n_largest_components:
        mask[np.where(np.logical_or(mask, components == lbl))] = 1
        
    return mask


def process_road_mask(img_labeled, n_largest=1):
    # Get the n largest connected components
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
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    roads_processed = cv2.morphologyEx(roads_processed, cv2.MORPH_CLOSE, kernel=ker)
    
    return roads_processed


def get_skeleton(mask):
    # Calculate number of neighbors with this kernel
    # 111
    # 101
    # 111
    ker_nb = np.ones((3, 3), dtype=np.uint8)
    ker_nb[1, 1] = 0
    
    removed_points = 1  # Just for initializing
    
    # While there are points to be removed, remove suitable border pixels
    while removed_points > 0:
        # Get the border
        # 1. Take the inverse mask
        mask_inv = np.ones_like(mask, dtype=np.uint8)
        mask_inv[np.where(mask == 1)] = 0
        
        # 2. Grow the inverse mask one pixel into the road-occupied space
        mask_inv = cv2.dilate(mask_inv, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=1)
        
        # 3. The overlap between both masks is the border (contour)
        border = np.logical_and(mask_inv, mask)
        
        # Count the number of neighbors of each border pixel
        border_filtered = cv2.filter2D(border.astype(np.uint8), 0, kernel=ker_nb)
        # Count the number of neighbors of each road pixel
        mask_filtered = cv2.filter2D(mask, 0, kernel=ker_nb)
        
        # Construct the part of the border that can be removed
        # - a pixel can be removed if it has exactly two border neighbors, but more than two road neighbors
        border_remove = border.copy()
        border_remove[np.where(np.logical_or(border_filtered != 2, mask_filtered <= border_filtered))] = 0
        
        # Set removed border pixels to 0
        mask_thinned = mask.copy()
        mask_thinned[np.where(border_remove == 1)] = 0
        # plt.clf()
        # plt.imshow(mask_thinned, cmap='gray')
        
        # Calculate the number of removed points
        removed_points = np.sum(np.logical_and(mask, mask_thinned == 0))
        mask = mask_thinned.copy()
        # plt.clf()
        # plt.imshow(mask, cmap='gray')
    
    # TODO: The diagonals could be one pixel thinner
    return mask


def thinning_wang(roads_thinned):
    # Algorithm as described in wang89
    roads_out = roads_thinned.copy()
    removed_points = 1
    while removed_points > 0:
        removed_points = 0
        contour = cv2.dilate((roads_out == 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
        contour = np.logical_and(contour, roads_out).astype(np.uint8)
        contour = np.pad(contour, (1, 1))
        index_i, index_j = np.where(contour)
        # plt.imshow(contour, cmap='gray')
    
        ker_nb = np.ones((3, 3), dtype=np.uint8)
        ker_nb[1, 1] = 0
    
        ker_c10 = np.array([[0, 1, 1], [0, 0, 1], [1, 0, 0]], dtype=np.uint8)
        ker_c12 = np.array([[0, 0, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
        ker_c20 = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 1]], dtype=np.uint8)
        ker_c22 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.uint8)
        
        c = lambda nbs: ((np.sum(nbs * ker_c10) == 0 and np.sum(nbs * ker_c12) == 2) or
                         (np.sum(nbs * ker_c20) == 0 and np.sum(nbs * ker_c22) == 2))
        
        neighbor_list = lambda mat: np.concatenate((mat.flatten()[:4], mat.flatten()[5:]))
        
        a = lambda nblist: np.sum(nblist != np.concatenate([nblist[1:], [nblist[0]]])) == 2
        
        g = 1
        for i, j in zip(index_i, index_j):
            neighbors = contour[i-1:i+2, j-1:j+2]
            number_of_neighbors = np.sum(neighbors * ker_nb)
            if number_of_neighbors < 2 or number_of_neighbors > 6:
                continue
            
            p = neighbor_list(neighbors)
            if not (c(neighbors) or a(p)):
                continue
            
            if g == 0 and not (p[2] + p[4] * p[0] * p[6] == 0):
                continue
            if g == 1 and not (p[0] + p[6] * p[2] * p[4] == 0):
                continue
            
            roads_out[i, j] = 0
            g = (g + 1) % 2
            removed_points += 1
    return roads_out
    
    # Vectorized version of wang89:
    # Unfortunately one process was not easy to vectorize, so this serves as reminder for a potential improvement
    # # ker_nb = np.ones((3, 3), dtype=np.uint8)
    # # ker_nb[1, 1] = 0
    # number_of_neighbors = cv2.filter2D(contour, 0, kernel=ker_nb)  # includes neighbors of non-contour pixels
    # contour_neighbors = np.logical_and(number_of_neighbors > 1, number_of_neighbors < 7)
    # # contour_neighbors[np.where(contour == 0)] = 0
    #
    # # ker_c10 = np.array([[0, 1, 1], [0, 0, 1], [1, 0, 0]], dtype=np.uint8)
    # # ker_c12 = np.array([[0, 0, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
    # # ker_c20 = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 1]], dtype=np.uint8)
    # # ker_c22 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.uint8)
    # filter_c10 = cv2.filter2D(contour, 0, ker_c10) == 0
    # filter_c12 = cv2.filter2D(contour, 0, ker_c12) == 2
    # c_either = np.logical_and(filter_c10, filter_c12)
    # filter_c20 = cv2.filter2D(contour, 0, ker_c20) == 0
    # filter_c22 = cv2.filter2D(contour, 0, ker_c22) == 2
    # c_or = np.logical_and(filter_c20, filter_c22)
    # contour_c = np.logical_or(c_either, c_or)  # includes neighbors of non-contour pixels
    # # contour_c[np.where(contour == 0)] = 0
    #
    # ker_cross_top1 = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=np.uint8)
    # ker_cross_top3 = np.array([[0, 0, 0], [1, 0, 1], [0, 1, 0]], dtype=np.uint8)
    # ker_cross_right1 = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=np.uint8)
    # ker_cross_right3 = np.array([[0, 1, 0], [1, 0, 0], [0, 1, 0]], dtype=np.uint8)
    # filter_top1 = cv2.filter2D(contour, 0, ker_cross_top1) == 0
    # filter_top3 = cv2.filter2D(contour, 0, ker_cross_top3) < 3
    # cross_either = np.logical_and(filter_top1, filter_top3)
    # filter_right1 = cv2.filter2D(contour, 0, ker_cross_right1) == 0
    # filter_right3 = cv2.filter2D(contour, 0, ker_cross_right3) < 3
    # cross_or = np.logical_and(filter_right1, filter_right3)
    # contour_cross = np.logical_or(cross_either, cross_or)  # includes neighbors of non-contour pixels
    # # contour_cross[np.where(contour == 0)] = 0
    #
    # to_be_removed = np.logical_and(contour_neighbors, np.logical_and(contour_c, contour_cross))
    # to_be_removed[np.where(contour == 0)] = 0
    #
    # plt.imshow(to_be_removed, cmap='gray')
    #
    # roads_thinned[to_be_removed] = 0
    # plt.imshow(roads_thinned, cmap='gray')


# Not used
def get_intersection_points(roads_thinned):
    # Get intersection points as described in jin12
    
    # Scan for points with at least 3 neighbors
    ker_eight_nb = np.ones((3, 3), dtype=np.uint8)
    ker_eight_nb[1, 1] = 0
    ker_four_nb = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    # neighbors_eight = cv2.filter2D(roads_thinned, 0, kernel=ker_eight_nb)
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
    c += np.where(np.logical_and(ker_bottom_left_cor, ker_bottom_left), 1, 0).astype(np.int32)
    ker_top_left = cv2.filter2D(roads_thinned, 0, kernel=np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])) == 0
    ker_top_left_cor = cv2.erode(roads_thinned, kernel=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.uint8))
    c += np.where(np.logical_and(ker_top_left_cor, ker_top_left), 1, 0).astype(np.int32)

    plt.clf()
    plt.imshow(c > 3, cmap='gray')

    return np.where(c > 3, 1, 0).astype(np.uint8)


if __name__ == '__main__':
    main()

