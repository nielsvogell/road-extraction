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


def plot_images(images, r, c, cmap=None, title=None):
    if title:
        plt.title(title)
    for index, image in enumerate(images):
        plt.subplot(r, c, index + 1)
        plt.axis('off')
        plt.imshow(image, cmap=cmap)
        # plt.tight_layout()
    plt.tight_layout()


def HSV2RGB(img_hsv):
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)


def BGR2RGB(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def show_color_evaluation(name):
    img = cv2.imread("data/" + name)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Cluster
    labels, cluster_colors = cluster_image_colors(img_rgb, nr_clusters=5)

    is_gray_l = lambda c: (1 - max([abs(c[1] - c[0]), abs(c[2] - c[1]), abs(c[0] - c[2])]) / 255) * 100
    for color in cluster_colors.values():
        # Evaluate Color
        evaluation, gray, red, green, blue = evaluate_color(color)
        
        title_rgb = "rgb color: {}\n".format(color)
        title_colors = "gray: {}%, red: {}%, green: {}%, blue: {}%\n".format(round(gray), round(red), round(green),
                                                                             round(blue))
        title_labels = "background: {}%, road: {}%, building: {}%".format(round(evaluation['background']),
                                                                          round(evaluation['road']),
                                                                          round(evaluation['building']))
        
        color_swatch_img = [[color, color, color, color, color, color]]
        
        plt.figure(figsize=(8, 8))
        plot_images([color_swatch_img], 1, 1, title=title_rgb + title_colors + title_labels)
    
    # copy img and change value of each pixel to the median value of its cluster
    # img_gm = copy.deepcopy(img_rgb)
    # index = 0
    # for x in range(len(img_gm)):
    #     for y in range(len(img_gm[x])):
    #         img_gm[x, y] = cluster_colors[labels[index]]
    #         index += 1
    
    # Arke: suggested optimization
    (h, w) = img_rgb.shape
    img_gm = np.array(cluster_colors[labels]).reshape(w, h).transpose()
    
    plt.figure(figsize=(30, 30))
    plot_images([img_gm], 1, 1)


def cluster_image_colors(img_rgb, nr_clusters):
    all_colors = [values for x in img_rgb.tolist() for values in x]
    labels, cluster_colors = gaussian_mixture_cluster(all_colors, nr_clusters)
    return labels, cluster_colors


# Arke: Just a comment, purely image process
def gaussian_mixture_cluster(x, nr_clusters, b_print=False, print_all_values=False):
    x = np.array(x)
    
    # Cluster with Gaussian Mixture
    gm = GaussianMixture(n_components=nr_clusters, random_state=0).fit(x)
    labels = gm.predict(x)
    
    unique_labels = np.unique(labels)
    n_clusters_ = len(unique_labels)
    
    # Calculate Median Value for each Cluster
    cluster_colors = {}
    
    if b_print:
        print("There are {} clusters!".format(n_clusters_))
    
    for k in unique_labels:
        my_members = labels == k
        if b_print:
            print("Label: {}".format(k))
        
        cluster_median_value = [0] * (len(x[0]))
        nr_points = 0
        
        for in_cluster, index in zip(my_members, range(len(my_members))):
            if in_cluster:
                if print_all_values:
                    print("\t{}".format(x[index]))
                nr_points += 1
                cluster_median_value = [x + y for x, y in zip(x[index], cluster_median_value)]
        
        cluster_colors[k] = [round(x / nr_points) for x in cluster_median_value]
        if b_print:
            print("Color: {}\n".format(cluster_colors[k]))
    
    return labels, cluster_colors


# returns building/background/road based on color assignment with "percent_gray_red_green_blue"
def evaluate_color(color_rgb):
    gray = is_gray(color_rgb, min(color_rgb) / 2)
    red, green, blue = is_red_green_blue(color_rgb)
    
    # gray -> road or building
    # green and red similar -> background
    
    background = (1 - (abs(red - green) / (red + green))) * 100
    not_background = (100 - background)
    
    not_gray = (100 - gray) / 100
    
    map_type = {'background': background * not_gray, 'road': gray / 2, 'building': not_background * not_gray + gray / 2}
    
    return map_type, gray, red, green, blue


# returns percentage on how gray
# based on grayish rgb colors have similar values for r, g, b
# Arke: Suggest to use 255 as diff_range. this gives similar results to std
#       Suggest to make this an array function
# Lambda equivalent:
# lambda c : (1 - max([abs(c[1] - c[0]), abs(c[2] - c[1]), abs(c[0] - c[2])]) / 255) * 100
def is_gray(color_rgb, diff_range=100):
    r, g, b = color_rgb
    diffs = abs(r - g), abs(r - b), abs(g - b)
    max_diff = max(diffs)
    gray_percentage = (1 - (max_diff / diff_range)) * 100
    return 0 if gray_percentage < 0 else gray_percentage

# Comparison between min-max approach and standard deviation
# colorspace = np.array([[r, g, b] for r in range(256) for b in range(256) for g in range(256)])
# grayspace = (1 - np.max(np.hstack((np.abs(colorspace[:, 1] - colorspace[:, 0])[:, np.newaxis],
#                                    np.abs(colorspace[:, 2] - colorspace[:, 1])[:, np.newaxis],
#                                    np.abs(colorspace[:, 0] - colorspace[:, 2])[:, np.newaxis])), axis=1) / 255) * 100
# grayspace[grayspace < 0] = 0
# max_std = 120.20815280171308
# grayspace_alt = (100 - np.std(colorspace, axis=1) / max_std * 100)
# print("Maximum difference: " + str(np.max(grayspace_alt - grayspace)) + ". Minimum difference: " +
#       str(np.min(grayspace_alt - grayspace)))


# returns array r, g, b of 0-100 percentage
# based on the r, g, b with the highest values and their diff to lower values
def is_red_green_blue(color_rgb, d=0):
    r, g, b = color_rgb
    
    # subtract half max value 
    # -> if lower value less than half of max value -> its too minimal to change color
    max_value = max([r, g, b])
    max_diff = int(max_value / 2)
    r, g, b = [(x - max_diff if x > max_diff else 0) for x in [r, g, b]]
    
    # subtract min value 
    # -> if the lower two values are similar it just grays the first color and doesn't change it
    min_value = min([r, g, b])
    r, g, b = [(x - min_value) for x in [r, g, b]]
    
    sum_value = r + g + b
    r, g, b = [(x / sum_value) * 100 if x > 0 else x for x in [r, g, b]]
    
    return [r, g, b]


img_name = "exp.png"
show_color_evaluation(img_name)
