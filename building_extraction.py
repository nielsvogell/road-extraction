# Imports
import cv2
import numpy as np
import sys
from segmentation import segment
if 'matplotlib' not in sys.modules:
    from matplotlib import use
    use('TkAgg')
from matplotlib import pyplot as plt


def main():
    print('Type the name of the input file.\n data/[munich.png]')
    file_name = input()
    if file_name == '':
        file_name = 'munich.png'
    image_name, extension = file_name.split('.')
    
    # Read image as RGB
    # Default: Load an example image from Google Maps
    img_rgb = cv2.cvtColor(cv2.imread(''.join(['data/', file_name])), cv2.COLOR_RGB2BGR)
    
    cluster_img, cluster_labels, cluster_colors = segment(img_rgb)
    
    detected_buildings = extract_buildings(cluster_img, cluster_labels, 120, 1500, 60,
                                           cluster_colors=cluster_colors,
                                           output_type="original", original=img_rgb)
    
    plt.imshow(detected_buildings)
    plt.axis('off')
    plt.savefig(''.join(['out/', image_name, '_buildings_out.png']), dpi=300)
    print('Saved output file to', ''.join(['out/', image_name, '_buildings_out.png']), '\n')


def process_labeled_img(cluster_img: np.ndarray, cluster_labels, cluster_colors=None, output_type='mask', original=None):
    background_mask = np.zeros_like(cluster_img, dtype=np.uint8)
    for lbl in cluster_labels['background']:
        background_mask[np.where(cluster_img == lbl)] = 255
        
    road_mask = np.zeros_like(cluster_img, dtype=np.uint8)
    for lbl in cluster_labels['road']:
        road_mask[np.where(cluster_img == lbl)] = 255
    
    # Blur image to make shapes stand out, remove noise & unnecessary details
    road_mask = cv2.bilateralFilter(road_mask, 12, 20, 500)
    
    # Ensure image is binary for morphological operations
    road_mask = cv2.threshold(road_mask, 101, 255, cv2.THRESH_BINARY)[1]
    
    # Remove secondary structures (large, stuctural noise)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5)),
                                 iterations=4)
    
    # Reconnect sections of horizontal & vertical roads that may be disconnected by shadows, bridges, etc...
    road_mask = cv2.dilate(road_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)), iterations=9)
    road_mask = cv2.dilate(road_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)), iterations=9)
    
    # Reconnect holes within roads to make uniform shapes
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                                 iterations=3)
    
    # Enhance features (to counteract dilation operations)
    road_mask = cv2.dilate(road_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=3)
    
    ##################################
    # Unused and potentially not working with current interfaces
    # # Make a mask with just road network in blue
    # roads = np.where(road_mask == [255, 255, 255], (0, 0, 255), 0)
    # # Make a mask with everything other than road network (=background+buildings) in green
    # background = np.where(road_mask == [0, 0, 0], (0, 255, 0), 0)
    # # Make a bicolor mask combining a blue road network & green background
    # background_and_roads = (background * 0.5 + roads * 0.5).astype(np.uint8)
    ##################################

    # Apply tricolor mask to original image
    (h, w) = cluster_img.shape
    color_eval_img = np.zeros((h, w, 3), dtype=np.uint8)
    if cluster_colors is not None:
        for lbl, color in cluster_colors.items():
            color_eval_img[np.where(cluster_img == lbl)] = color.astype(np.uint8)
    else:
        # TODO: Choose better colors for distinguishing later
        default_colors = {'background': np.array([206, 234, 214], dtype=np.uint8),
                          'road': np.array([241, 243, 244], dtype=np.uint8),
                          'building': np.array([252, 232, 230], dtype=np.uint8)}
        
        for tp, color in default_colors.items():
            for lbl in cluster_labels[tp]:
                color_eval_img[np.where(cluster_img == lbl)] = color
    
    # Generating output
    if output_type == 'original' and original is not None:
        output = original.copy().astype(np.uint8)
    elif output_type == 'label_color':
        output = color_eval_img.copy().astype(np.uint8)
    else:  # output_type == 'mask'
        output = np.zeros_like(cluster_img).astype(np.uint8)
    
    return output, color_eval_img


def extract_buildings(cluster_img: np.ndarray, cluster_labels: dict,
                      thresh=120, min_building_area=1500, nb_buildings=None, cluster_colors=None,
                      output_type='mask', original: np.ndarray = None):
    """
    
    :param cluster_img: Labeled image returned by the segmentation algorithm
    :param cluster_labels: dictionary containing the labels associated with roads, buildings and background
    :param thresh: threshold for corner matching
    :param min_building_area: min area in pixels required for a contour to be considered a building
    :param nb_buildings: expected number of buildings (only internal usage)
    :param cluster_colors: colors assigned to each label returned by the segmentation
    :param output_type: choice of 'mask' (empty canvas), 'original' (input image), 'label_color' (segmentation result)
    :param original: input image to draw on
    :return:
    """
    output, color_img = process_labeled_img(cluster_img, cluster_labels,
                                            cluster_colors=cluster_colors, output_type=output_type,
                                            original=original)
    max_nr_buildings = 0
    
    building_mask = np.zeros_like(color_img, dtype=np.uint8)
    for lbl in cluster_labels['building']:
        lbl_indices = np.where(cluster_img == lbl)
        building_mask[lbl_indices] = color_img[lbl_indices]
        
    output, nr_buildings = extract_buildings_with_label(building_mask, output.copy(), color_img,
                                                        thresh=thresh,
                                                        min_building_area=min_building_area,
                                                        nb_buildings=nb_buildings)

    return output
    

def extract_buildings_with_label(building_mask: np.ndarray, output: np.ndarray,
                                 color_img: np.ndarray,
                                 thresh: int = 120, min_building_area=1500, nb_buildings=None):
    """
    Detects contours of buildings in the image and display an accuracy metric
    @Author : Tanguy Gerniers (W21)
    @Args :
      building_mask (np.ndarray) : Mask of segmented image showing only labels associated with buildings
      output (np.ndarray) : The prepared output array to draw on.
      color_img (np.ndarray) : Segmentation result with median color for each label associated with buildings
      tresh (int) : Determines whether contours should follow contour edges (0) or bound them within smoother
                    rectangular boxes where possible (120)
      min_building_area(int) : minimum area a building must be to be considered (recommended : 500-1500, depending on
                               height from which image is taken)
      nb_buildings(int) : number of buildings effectively in the image (optional)

    @Returns :      output (image) : background of choice with contours of detected buildings in red applied on top as a
                                     mask
    """
    # Blur image to make shapes stand out, remove noise & unnecessary details
    buildings_processed = cv2.bilateralFilter(building_mask.copy(), 12, 20, 500)
    
    # Ensure image is binary for morphological operations
    buildings_processed = cv2.threshold(buildings_processed, 101, 255, cv2.THRESH_BINARY)[1]
    
    # Remove secondary structures (large, stuctural noise such as trees, pieces of road that slipped through, etc..)
    buildings_processed = cv2.morphologyEx(buildings_processed, cv2.MORPH_CLOSE,
                                           cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=2)
    
    # Reconnect holes within structures to make uniform shapes
    buildings_processed = cv2.morphologyEx(buildings_processed, cv2.MORPH_OPEN,
                                           cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=2)
    
    # Increase structure sizes to merge disconnected portions together
    buildings_processed = cv2.dilate(buildings_processed,
                                     cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=3)
    
    # Detect building edges
    edges = cv2.Canny(buildings_processed, 110, 255)
    
    # Extract contours & hierarchy from edges
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Prepare output
    # output, color_eval_img = process_labeled_img(cluster_img, label_colors, output_type, original)
    
    # list storing contours that are not rectangular
    unique_contours = []
    
    counter = 0
    index = -1
    # for each contour (potential building)
    for contour in contours:
        index += 1
        min_x = 10000
        max_x = 0
        min_y = 10000
        max_y = 0
        
        # extract positional information of contour
        for point in contour:
            
            if point[0][0] < min_x:
                min_x = point[0][0]
            if point[0][0] > max_x:
                max_x = point[0][0]
            if point[0][1] < min_y:
                min_y = point[0][1]
            if point[0][1] > max_y:
                max_y = point[0][1]
        dx = abs(max_x - min_x)
        dy = abs(max_y - min_y)
        
        # if contour is not a road (overly long and thin structure) and not within another contour:
        if dx < 3.5 * dy and dy < 3.5 * dx and hierarchy[0][index][3] == -1 and ((dx * dy) > min_building_area):
            
            # records number of contour candidates that are kept to measure accuracy
            counter += 1
            
            # Get dimensions of general area of contour
            x, y, w, h = cv2.boundingRect(contour)
            
            # Compute average color of top/bottom left/right corners of the contour
            # avg_color_cnt = np.array(cv2.mean(color_eval_img[y:(y + h), x:(x + w)])).astype(np.uint8)
            # d0 = int(avg_color_cnt[0]) + int(avg_color_cnt[1]) + int(avg_color_cnt[2])
            # bottom left
            avg_color1 = np.array(cv2.mean(color_img[y: y + int(h / 2), x: x + int(w / 2)])).astype(np.uint8)
            d1 = int(avg_color1[0]) + int(avg_color1[1]) + int(avg_color1[2])
            # top left
            avg_color2 = np.array(cv2.mean(color_img[y + int(h / 2): y + h, x: x + int(w / 2)])).astype(np.uint8)
            d2 = int(avg_color2[0]) + int(avg_color2[1]) + int(avg_color2[2])
            # bottom right
            avg_color3 = np.array(cv2.mean(color_img[y: y + int(h / 2), x + int(w / 2): x + w])).astype(np.uint8)
            d3 = int(avg_color3[0]) + int(avg_color3[1]) + int(avg_color3[2])
            # top right
            avg_color4 = np.array(cv2.mean(color_img[y + int(h / 2): y + h, x + int(w / 2): x + w])).astype(
                np.uint8)
            d4 = int(avg_color4[0]) + int(avg_color4[1]) + int(avg_color4[2])
            
            # If corners of contour are sufficiently similar to one another, it is probably a rectangle/trapezoid
            #   -> draw smooth & rectangular bounding box
            if abs(d1 - d2) < thresh and abs(d1 - d3) < thresh and abs(d1 - d4) < thresh and abs(d2 - d3) < thresh and \
                    abs(d2 - d4) < thresh and abs(d3 - d4) < thresh:
                # Draw rectangular building contours, in red, onto chosen background
                if len(output.shape) == 3:
                    output = cv2.line(output, (min_x, min_y), (min_x, max_y), (255, 0, 0), 3)
                    output = cv2.line(output, (min_x, min_y), (max_x, min_y), (255, 0, 0), 3)
                    output = cv2.line(output, (max_x, max_y), (min_x, max_y), (255, 0, 0), 3)
                    output = cv2.line(output, (max_x, max_y), (max_x, min_y), (255, 0, 0), 3)
                else:
                    output = cv2.line(output, (min_x, min_y), (min_x, max_y), 255, 3)
                    output = cv2.line(output, (min_x, min_y), (max_x, min_y), 255, 3)
                    output = cv2.line(output, (max_x, max_y), (min_x, max_y), 255, 3)
                    output = cv2.line(output, (max_x, max_y), (max_x, min_y), 255, 3)
                
            # If corners of contour are sufficiently different to one another, it probably has a unique shape
            #     -> draw contour as it is
            else:
                unique_contours.append(contour)
    
    if nb_buildings:
        print("Accuracy score :", str((counter / nb_buildings) * 100)[:4], "%",
              " (approximately %i out of %i buildings have been detected)" % (counter, nb_buildings))

    # Draw unique building contours, in red,  onto chosen background
    if len(unique_contours) > 0:
        output = cv2.drawContours(output, unique_contours, -1, (255, 0, 0), 3)
        
    return output, len(contours)


if __name__ == '__main__':
    main()
    