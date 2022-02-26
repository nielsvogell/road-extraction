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



def building_detection(labeled_img, tresh, min_building_area, nb_buildings=None, output_type='mask', original=None):
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
    
    
    ### Common code
    if output_type == 'original' and original:
        output = original.copy()
    elif output_type == 'tricolor':
        # Apply tricolor mask to original image
        (h, w) = labeled_img.shape
        output = np.zeros((h, w, 3), dtype=np.uint8)
        output[np.where(labeled_img == labels['background'])] = np.array([206, 234, 214], dtype=np.uint8)
        output[np.where(road_mask)] = np.array([241, 243, 244], dtype=np.uint8)
        output[np.where(output == [0, 0, 0])] = np.array([252, 232, 230], dtype=np.uint8)
    else: # output_type == 'mask'
        output = np.zeros_like(labeled_img)
    
    try:
        building_mask = color_eval_img == labels['building']
    
    except:
        imagea = np.where(image == [120, 110, 128], 255, 0)
        imageb = np.where(image == [42, 48, 66], 255, 0)
        imagec = np.where(image == [156, 144, 14], 255, 0)
        building_mask = np.logical_or(imagea, np.logical_or(imageb, imagec)) * 255
        building_mask = building_mask.astype(np.uint8)
    
    # Blur image to make shapes stand out, remove noise & unnecessary details
    filtered = cv2.bilateralFilter(building_mask, 12, 20, 500)
    
    # Ensure image is binary for morphological operations
    binary = cv2.threshold(filtered, 101, 255, cv2.THRESH_BINARY)[1]
    
    # Remove secondary structures (large, stuctural noise such as trees, pieces of road that slipped through, etc..)
    morph1 = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=2)
    
    # Reconnect holes within structures to make uniform shapes
    morph2 = cv2.morphologyEx(morph1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=2)
    
    # Increase structure sizes to merge disconnected portions together
    morph3 = cv2.dilate(morph2, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=3)
    
    # Detect building edges
    edges = cv2.Canny(morph3, 110, 255)
    
    # Extract contours & hierarchy from edges
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # list storing contours that are not rectangular
    unique_contours = []
    
    counter = 0
    index = -1
    # for each contour (potential building)
    for contour in contours:
        index += 1
        minx = 10000
        maxx = 0
        miny = 10000
        maxy = 0
        dx = 0
        dy = 0
        # extract positional information of contour
        for point in contour:
            
            if point[0][0] < minx:
                minx = point[0][0]
            if point[0][0] > maxx:
                maxx = point[0][0]
            if point[0][1] < miny:
                miny = point[0][1]
            if point[0][1] > maxy:
                maxy = point[0][1]
        dx = abs(maxx - minx)
        dy = abs(maxy - miny)
        
        # if contour is not a road (overly long and thin structure) and not within another contour:
        if dx < 3.5 * dy and dy < 3.5 * dx and hierarchy[0][index][3] == -1 and ((dx * dy) > min_building_area):
            
            # records number of contour candidates that are kept to measure accuracy
            counter += 1
            
            # Get dimensions of general area of contour
            x, y, w, h = cv2.boundingRect(contour)
            
            # Compute average color of top/bottom left/right corners of the contour
            avg_color_cnt = np.array(cv2.mean(color_eval_img[y: y + h, x: x + w])).astype(np.uint8)
            d0 = int(avg_color_cnt[0]) + int(avg_color_cnt[1]) + int(avg_color_cnt[2])
            avg_color1 = np.array(cv2.mean(color_eval_img[y: y + int(h / 2), x: x + int(w / 2)])).astype(np.uint8)
            d1 = int(avg_color1[0]) + int(avg_color1[1]) + int(avg_color1[2])
            avg_color2 = np.array(cv2.mean(color_eval_img[y + int(h / 2): y + h, x: x + int(w / 2)])).astype(np.uint8)
            d2 = int(avg_color2[0]) + int(avg_color2[1]) + int(avg_color2[2])
            avg_color3 = np.array(cv2.mean(color_eval_img[y: y + int(h / 2), x + int(w / 2): x + w])).astype(np.uint8)
            d3 = int(avg_color3[0]) + int(avg_color3[1]) + int(avg_color3[2])
            avg_color4 = np.array(cv2.mean(color_eval_img[y + int(h / 2): y + h, x + int(w / 2): x + w])).astype(
                np.uint8)
            d4 = int(avg_color4[0]) + int(avg_color4[1]) + int(avg_color4[2])
            
            # If corners of contour are sufficiently similar to one another, it is probably a rectangle/trapezoid -> draw smooth & rectangular bounding box
            if abs(d1 - d2) < tresh and abs(d1 - d3) < tresh and abs(d1 - d4) < tresh and abs(d2 - d3) < tresh and abs(
                    d2 - d4) < tresh and abs(d3 - d4) < tresh:
                # Draw rectangular building contours, in red, onto chosen background
                output = cv2.line(output, (minx, miny), (minx, maxy), (255, 0, 0), 3)
                output = cv2.line(output, (minx, miny), (maxx, miny), (255, 0, 0), 3)
                output = cv2.line(output, (maxx, maxy), (minx, maxy), (255, 0, 0), 3)
                output = cv2.line(output, (maxx, maxy), (maxx, miny), (255, 0, 0), 3)
            
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