# Imports
import cv2
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import copy
from PIL import Image
from enum import Enum
from scipy import spatial
from sklearn.mixture import GaussianMixture
import imutils
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.interpolate import splprep, splev
from scipy import ndimage
import pickle as pcl
import time

import segmentation
from segmentation import segment


def main():
    image_path = 'data/munich.png'
    
    labeled_img, labels = segment(image_path)
    
    image = labeled_img.copy()
    
    ### output_type : "mask"
    
    mask =
    
    ### output_type : "tricolor"
    
    try:
    
    
    except:
        imagea = np.where(image == [170, 167, 175], 255, 0)
        imageb = np.where(image == [48, 54, 67], 0, 0)
        road_mask = (imagea * 0.5 + imageb * 0.5).astype(np.uint8)
        road_mask = road_mask.astype(np.uint8)

    detected_buildings = building_detection(labeled_img, 120, 1500, 60, output_type="original")


def morph2(img, kernel):
    (h, w) = img.shape[:2]  # get image dimensions
    
    for i in range(10):
        Im_d = cv2.dilate(img, kernel, iterations=1)  # dilate
        Im_e = cv2.erode(img, kernel, iterations=1)  # erode
        Im_h = 0.5 * (Im_d + Im_e)  # combination of dilate and erode
        
        # if original pixel is darker than dilated and eroded image, keep dilated image else the eroded image
        img = np.where(img > Im_h, Im_d, Im_e)
    
    return img


def morph(img, kernel):
    (h, w) = img.shape[:2]  # get image dimensions
    
    for i in range(10):
        Im_d = cv2.dilate(img, kernel, iterations=1)  # dilate
        Im_e = cv2.erode(img, kernel, iterations=1)  # erode
        Im_h = 0.5 * (Im_d + Im_e)  # combination of dilate and erode
        
        for y in range(0, h):
            for x in range(0, w):
                # threshold the pixel
                if img[y, x] > Im_h[y, x]:
                    img[y, x] = Im_d[y, x]  # if original pixel is darker than dilated and eroded pixel, dilate pixel
                else:
                    img[y, x] = Im_e[y, x]  # if original pixel is lighter than dilated and eroded pixel, erode pixel
    
    return img



def process_labeled_img(labeled_img, label_colors=None, output_type='mask', original=None):
    labels = segmentation.get_labels()
    road_mask = labeled_img == labels['road']
    
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
    
    # Make a mask with just road network in blue
    roads = np.where(road_mask == [255, 255, 255], (0, 0, 255), 0)
    # Make a mask with everything other than road network (=background+buildings) in green
    background = np.where(road_mask == [0, 0, 0], (0, 255, 0), 0)
    # Make a bicolor mask combining a blue road network & green background
    background_and_roads = (background * 0.5 + roads * 0.5).astype(np.uint8)

    # Apply tricolor mask to original image
    (h, w) = labeled_img.shape
    if label_colors:
        col_bg = label_colors['background']
        col_rd = label_colors['road']
        col_bd = label_colors['building']
    else:
        col_bg = np.array([206, 234, 214], dtype=np.uint8)
        col_rd = np.array([241, 243, 244], dtype=np.uint8)
        col_bd = np.array([252, 232, 230], dtype=np.uint8)
        
    color_eval_img = np.zeros((h, w, 3), dtype=np.uint8)
    color_eval_img[np.where(labeled_img == labels['background'])] = col_bg
    color_eval_img[np.where(road_mask)] = col_rd
    color_eval_img[np.where(color_eval_img == [0, 0, 0])] = col_bd

    ### Generating output
    if output_type == 'original' and original:
        output = original.copy()
    elif output_type == 'tricolor':
        output = color_eval_img.copy()
    else:  # output_type == 'mask'
        output = np.zeros_like(labeled_img)
    
    return output, color_eval_img


def building_detection(labeled_img, tresh, min_building_area, nb_buildings=None, label_colors=None, output_type='mask',
                       original=None):
    """ Detects contours of buildings in the image and display an accuracy metric
    @Author : Tanguy Gerniers (W21)
    @Args :
      path (str) : The file location of the image.
      output_type(str) :  "mask" : no background
                          "original" : original image as background
                          "tricolor" : 2 clusters (road network in blue, everything else in green) as background
      tresh (int) : Determines whether contours should follow contour edges (0) or bound them within smoother
                    rectangular boxes where possible (120)
      min_building_area(int) : minimum area a building must be to be considered (recommended : 500-1500, depending on
                               height from which image is taken)
      nb_buildings(int) : number of buildings effectively in the image (optional)

    @Returns :      output (image) : background of choice with contours of detected buildings in red applied on top as a
                                     mask
    """
    
    labels = segmentation.get_labels()
    building_mask = labeled_img == labels['building']
    
    # Blur image to make shapes stand out, remove noise & unnecessary details
    buildings_processed = cv2.bilateralFilter(building_mask, 12, 20, 500)
    
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
    output, color_eval_img = process_labeled_img(labeled_img, label_colors, output_type, original)
    
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
            avg_color1 = np.array(cv2.mean(color_eval_img[y: y + int(h / 2), x: x + int(w / 2)])).astype(np.uint8)
            d1 = int(avg_color1[0]) + int(avg_color1[1]) + int(avg_color1[2])
            # top left
            avg_color2 = np.array(cv2.mean(color_eval_img[y + int(h / 2): y + h, x: x + int(w / 2)])).astype(np.uint8)
            d2 = int(avg_color2[0]) + int(avg_color2[1]) + int(avg_color2[2])
            # bottom right
            avg_color3 = np.array(cv2.mean(color_eval_img[y: y + int(h / 2), x + int(w / 2): x + w])).astype(np.uint8)
            d3 = int(avg_color3[0]) + int(avg_color3[1]) + int(avg_color3[2])
            # top right
            avg_color4 = np.array(cv2.mean(color_eval_img[y + int(h / 2): y + h, x + int(w / 2): x + w])).astype(
                np.uint8)
            d4 = int(avg_color4[0]) + int(avg_color4[1]) + int(avg_color4[2])
            
            # If corners of contour are sufficiently similar to one another, it is probably a rectangle/trapezoid
            #   -> draw smooth & rectangular bounding box
            if abs(d1 - d2) < tresh and abs(d1 - d3) < tresh and abs(d1 - d4) < tresh and abs(d2 - d3) < tresh and abs(
                    d2 - d4) < tresh and abs(d3 - d4) < tresh:
                # Draw rectangular building contours, in red, onto chosen background
                output = cv2.line(output, (min_x, min_y), (min_x, max_y), (255, 0, 0), 3)
                output = cv2.line(output, (min_x, min_y), (max_x, min_y), (255, 0, 0), 3)
                output = cv2.line(output, (max_x, max_y), (min_x, max_y), (255, 0, 0), 3)
                output = cv2.line(output, (max_x, max_y), (max_x, min_y), (255, 0, 0), 3)
            
            # If corners of contour are sufficiently different to one another, it probably has a unique shape -> draw contour as it is
            else:
                unique_contours.append(contour)
    
    if nb_buildings:
        print("Accuracy score :", str((counter / nb_buildings) * 100)[:4], "%",
              " (approximately %i out of %i buildings have been detected)" % (counter, nb_buildings))

    
    # Draw unique building contours, in red,  onto chosen background
    output = cv2.drawContours(output, unique_contours, -1, (255, 0, 0), 3)
    
    return output




if __name__ == '__main__':
    main()