# Imports
import cv2
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import copy
from PIL import Image
from enum import Enum
from scipy import spatial
from sklearn.mixture import GaussianMixture

import time

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

    for color in cluster_colors.values():
        # Evaluate Color
        evaluation, gray, red, green, blue = evaluate_color(color)
        
        title_rgb = "rgb color: {}\n".format(color)
        title_colors = "gray: {}%, red: {}%, green: {}%, blue: {}%\n".format(round(gray), round(red), round(green),
                                                                             round(blue))
        title_labels = "background: {}%, road: {}%, building: {}%".format(round(evaluation['background']),
                                                                          round(evaluation['road']),
                                                                          round(evaluation['building']))
        
        color_swatch_img = [[color]*6]
        
        # TODO subplots
        plt.figure(figsize=(8, 8))
        plot_images([color_swatch_img], 1, 1, title=title_rgb + title_colors + title_labels)
    
    # copy img and change value of each pixel to the median value of its cluster
    # img_gm = copy.deepcopy(img_rgb)
    # index = 0
    # for x in range(len(img_gm)):
    #     for y in range(len(img_gm[x])):
    #         img_gm[x, y] = cluster_colors[labels[index]]
    #         index += 1

    # Arke: suggested optimization (removes the 1000 x 1000 loop and only iterates through the clusters)
    (h, w, d) = img_rgb.shape
    label_img = labels.reshape(h, w)
    img_gm = np.zeros_like(img_rgb)
    for lbl, color in cluster_colors.items():
        img_gm[np.where(label_img == lbl)] = color
    
    plt.figure(figsize=(15, 15))
    plot_images([img_gm], 1, 1)

# Arke: Optimized by reshaping
def cluster_image_colors(img_rgb, nr_clusters):
    all_colors = img_rgb.reshape((-1, 3))
    labels, cluster_colors = gaussian_mixture_cluster(all_colors, nr_clusters)
    return labels, cluster_colors


# Arke: Just a comment, this takes a lot of time. It might be a great method, but finding a faster one is desirable.
def gaussian_mixture_cluster(colors, nr_clusters, b_print=False, print_all_values=False):
    colors = np.array(colors)
    
    # Cluster with Gaussian Mixture
    gm = GaussianMixture(n_components=nr_clusters, random_state=0).fit(colors)
    labels = gm.predict(colors)
    
    unique_labels = np.unique(labels)
    n_clusters_ = len(unique_labels)
    
    # Calculate Median Value for each Cluster
    cluster_colors = {}
    
    if b_print:
        print("There are {} clusters!".format(n_clusters_))
    
    for k in unique_labels:
        b_labeled_k = labels == k
        if b_print:
            print("Label: {}".format(k))
        
        # cluster_median_value = [0] * (len(x[0]))
        # nr_points = 0
        #
        # for index, in_cluster in enumerate(b_labeled_k):
        #     if in_cluster:
        #         if print_all_values:
        #             print("\t{}".format(x[index]))
        #         nr_points += 1
        #         cluster_median_value = [x + y for x, y in zip(x[index], cluster_median_value)]
        #
        # nr_points = np.count_nonzero(b_labeled_k)  # Gets the number of points faster
        # cluster_colors[k] = [round(x / nr_points) for x in cluster_median_value]
        # # I think you are calculating the mean
        #
        # Arke: Suggest to optimize by directly calculating the median from the set of colors labeled k
        cluster_colors[k] = np.median(colors[b_labeled_k], axis=0).astype(np.uint8)
        if b_print:
            print("Color: {}\n".format(cluster_colors[k]))
    
    return labels, cluster_colors


# returns building/background/road based on color assignment with "percent_gray_red_green_blue"
def evaluate_color(color_rgb):
    # Arke: Possible code simplification:
    # is_gray could be a one-liner, removing the need for a function definition
    # Probably not faster and not necessarily more readable
    # is_gray = lambda c: (1 - max([abs(c[1] - c[0]), abs(c[2] - c[1]), abs(c[0] - c[2])]) / 255) * 100
    
    gray = is_gray(color_rgb, min(color_rgb) / 2)  # Why the min_color/2?
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
# Arke: Suggesting to use 255 as diff_range. this gives similar results to standard deviation
#       Suggesting to make this an array function (but for small cluster number the performance gain is not significant)
# Lambda equivalent:
# lambda c : (1 - max([abs(c[1] - c[0]), abs(c[2] - c[1]), abs(c[0] - c[2])]) / 255) * 100
def is_gray(color_rgb, diff_range=100):
    r, g, b = color_rgb
    diffs = abs(r - g), abs(r - b), abs(g - b)
    max_diff = max(diffs)
    gray_percentage = (1 - (max_diff / diff_range)) * 100
    return 0 if gray_percentage < 0 else gray_percentage

# Arke:
# Comparison between min-max approach and standard deviation
# colorspace = np.array([[r, g, b] for r in range(256) for b in range(256) for g in range(256)])
# t_start = time.time()
# grayspace = (1 - np.max(np.hstack((np.abs(colorspace[:, 1] - colorspace[:, 0])[:, np.newaxis],
#                                    np.abs(colorspace[:, 2] - colorspace[:, 1])[:, np.newaxis],
#                                    np.abs(colorspace[:, 0] - colorspace[:, 2])[:, np.newaxis])), axis=1) / 255) * 100
# grayspace[grayspace < 0] = 0
# t_end = time.time()
# print("Time to calculate min-max score: " + str(t_end - t_start))
# max_std = 120.20815280171308
# t_start = time.time()
# grayspace_alt = (100 - np.std(colorspace, axis=1) / max_std * 100)
# t_end = time.time()
# print("Time to calculate standard deviation: " + str(t_end - t_start))
# print("Maximum difference: " + str(np.max(grayspace_alt - grayspace)) + ". Minimum difference: " +
#       str(np.min(grayspace_alt - grayspace)))
# Conclusion: Standard deviation intuitively makes more sense to me and is more readable, but takes longer to calculate.


# returns array r, g, b of 0-100 percentage
# based on the r, g, b with the highest values and their diff to lower values
def is_red_green_blue(color_rgb, d=0):
    # Arke: Comment - There is no need to split rgb value and also not for the for loops.
    # Only really bad for vectorization of the function. For single-value calculation almost no difference.
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

    # Arke: Shorter, but not significantly faster. But vectorized, so it works for arrays of colors
    # c = np.array(color_rgb).reshape((-1, 3))  # make sure color is a n x 3 numpy array
    # c_max = np.max(c, axis=1)
    # c_min = np.min(c, axis=1)
    # c_sum = np.sum(c, axis=1)[:, np.newaxis]
    # c_sub = np.maximum((c_max / 2), c_min)[:, np.newaxis]
    # tmp = np.maximum(c - c_sub, 0)
    # return np.divide(tmp2, (c_sum - c_sub)) * 100
    return [r, g, b]

# Arke: Discussing vectorization
# It would be significantly faster to simultaneously evaluate all colors. However, with only a handful of clusters,
# the gain is not that big.
#
# c = np.tile(np.array([[120, 230, 60]]), (1000, 1))
# t_start = time.time()
# tmp1 = np.zeros((1000, 3))
# for i in range(1000):
#     tmp1[i,:] = is_red_green_blue(c[i,:])
# t_end = time.time()
# print(t_end - t_start)
# t_start = time.time()
# c_max = np.max(c, axis=1)
# c_min = np.min(c, axis=1)
# c_sum = np.sum(c, axis=1)
# # c_mid = c_sum - c_max - c_min
# c_sub = np.maximum((c_max / 2), c_min)[:, np.newaxis]
# tmp2 = np.maximum(c - np.maximum((c_max / 2), c_min)[:, np.newaxis], 0)
# tmp2 = np.divide(tmp2, np.sum(tmp2, axis=1)[:, np.newaxis]) * 100
# t_end = time.time()
# print(t_end - t_start)

img_name = "exp.png"
show_color_evaluation(img_name)
