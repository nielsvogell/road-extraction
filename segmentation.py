import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
import sys
if 'matplotlib' not in sys.modules:
    from matplotlib import use
    use('TkAgg')
from matplotlib import pyplot as plt


def main():
    test_segment()


def test_segment():
    print('Type the name of the input file.\n data/[munich.png]')
    file_name = input()
    if file_name == '':
        file_name = 'munich.png'
    image_name, extension = file_name.split('.')
    
    # read image
    img = cv2.imread(''.join(['data/', file_name]))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # SEGMENT
    cluster_img, cluster_labels, cluster_colors = segment(img_rgb)

    # plot output
    plt.figure(figsize=(10, 10))

    # 'background': 0, 'road': 1, 'building': 2

    # 'background', 'road', 'building'
    for index, map_type in enumerate(cluster_labels.keys()):
        mask = np.zeros_like(cluster_img)
        for lbl in cluster_labels[map_type]:
            mask[np.where(cluster_img == lbl)] = 1
        plt.subplot(2, 2, index+1)
        plt.axis('off')
        plt.imshow(mask, cmap="gray")
        plt.title(map_type, fontsize=18)

    plt.tight_layout()
    plt.savefig(''.join(['out/', image_name, '_segmentation_out.png']), dpi=300)
    print('Saved output file to', ''.join(['out/', image_name, '_segmentation_out.png']), '\n')


def get_labels(label='all'):
    labels = {'background': 0, 'road': 1, 'building': 2}
    if label == 'all':
        return labels
    elif label in labels.keys():
        return labels[label]
    else:
        raise 'Error:UnknownLabel'


# calls cluster function and evaluate color function for each clustered color
# plots the result
def show_color_evaluation(img_path, scale=0.3, blur_size=7, nr_clusters=5):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Cluster colors of image
    labels, cluster_colors = gaussian_mixture_cluster(img_rgb, blur_size=blur_size, scale=scale,
                                                      nr_clusters=nr_clusters)

    # PLOT
    plt.figure(figsize=(10, nr_clusters*2.5))

    # Iterate all cluster colors
    for index, color in enumerate(cluster_colors.values()):
        # Evaluate the color by getting color appearance and matching a type to each color appearance
        map_type, color_appearance = evaluate_color(color)

        # PLOT every color

        # Format every percentage float to string (0.25 -> "25%")
        for key, value in map_type.items():
            new_value = "{}%".format(round(value*100))
            map_type[key] = new_value

        for key, value in color_appearance.items():
            new_value = "{}%".format(round(value*100))
            color_appearance[key] = new_value

        # get titles
        title_rgb = "rgb color: {}\n".format(color)
        title_colors = str(color_appearance) + "\n"
        title_labels = str(map_type) + "\n"
        # make color swatch image
        color_swatch_img = [[color] * 9]

        plt.subplot(nr_clusters, 1, index+1)
        plt.axis('off')
        plt.imshow(color_swatch_img)
        plt.title(title_rgb + title_colors + title_labels, fontsize=18)

    plt.tight_layout()

    # reshapes one dimensional color array into the two image dimensions
    h, w, _ = img_rgb.shape
    label_img = labels.reshape(h, w)
    # makes "empty" image with same dimensions as img_rgb (original)
    img_gm = np.zeros_like(img_rgb)
    # for every color, change labeled pixels of image to this color
    for lbl, color in cluster_colors.items():
        img_gm[np.where(label_img == lbl)] = color

    # PLOT segmented image
    plt.figure(figsize=(10, 7))
    plt.subplot(1, 1, 1)
    plt.suptitle("Segmented Image", fontsize=18)
    plt.axis('off')
    plt.imshow(img_gm)

    plt.tight_layout()
    plt.show()


# calls gaussian mixture clustering and evaluate color function for each color
# maps most probable label ('background' 0, 'road' 1, 'building' 2) for each pixel
# input image, resizing scale, blur scale, and number clusters
# returns two dim array of image shape with one of following labels:
#    'background': 0, 'road': 1, 'building': 2 for each pixel
def segment(img_rgb, scale=0.3, blur_size=7, nr_clusters=5):
    # calls gaussian mixture clustering
    cluster_img, cluster_colors = gaussian_mixture_cluster(img_rgb, blur_size=blur_size, scale=scale,
                                                              nr_clusters=nr_clusters)

    cluster_labels = {'building': [], 'road': [], 'background': []}
    for lbl, color in cluster_colors.items():
        # evaluate color returns ('background', 'road', 'building') with percentage probability for each label
        map_type_probability, _ = evaluate_color(color)
        # get the label with max probability
        max_map_type = max(map_type_probability, key=map_type_probability.get)
        # store all labels that belong to each map type
        cluster_labels[max_map_type].append(lbl)

    # reshape one dimensional image into two dimensions of original image
    h, w, _ = img_rgb.shape
    cluster_img = cluster_img.reshape(h, w)

    return cluster_img, cluster_labels, cluster_colors


# CLUSTERS ALL COLORS OF THE IMAGE
# input rgb image, the blur factor, the scale factor and number of cluster
# returns labels -> list of all cluster labels for every pixel, cluster_colors -> list of colors for each cluster label
def gaussian_mixture_cluster(img_rgb, blur_size=0, scale=None, nr_clusters=5):
    # uses median blur on image to get rid of noise
    if blur_size > 0:
        img_rgb = cv2.medianBlur(img_rgb, blur_size)

    # get one dimensional array of all colors
    colors_train = colors = img_rgb.reshape((-1, 3))

    # use colors of resized image (to given scale) to train the gaussian mixture
    # -> faster clustering without loosing color information
    if scale:
        height, width = img_rgb.shape[:2]
        new_dim = (int(width * scale), int(height * scale))
        img_rgb_resized = cv2.resize(img_rgb, new_dim, interpolation=cv2.INTER_AREA)
        # get one dimensional array of all colors of resized image
        colors_train = img_rgb_resized.reshape((-1, 3))

    # cluster colors with Gaussian Mixture
    gm = GaussianMixture(n_components=nr_clusters, random_state=0).fit(colors_train)

    # predict the labels on the colors from the original image (not resized)
    # labels lists the label of the cluster for every pixel color in the image
    labels = gm.predict(colors)

    # calculate the median color value for every cluster
    cluster_colors = {}
    for k in np.unique(labels):
        # list where all pixels with the label k are True and rest False
        b_labeled_k = labels == k
        # calculates median of colors for where pixels are True
        # colors as type int32 -> so further calculations can go negative
        cluster_colors[k] = np.mean(colors[b_labeled_k], axis=0).astype(np.int32)

    return labels, cluster_colors


# evaluates a type (background, road, building) to each color
# input a rgb color; return the percentage for each map type and the color appearance
def evaluate_color(color_rgb):
    # get appearance of gray, red, green, blue
    gray, red, green, blue = get_color_appearance(color_rgb)

    if (red + green) == 0:
        # don't divide by zero -> set it to 1
        red_green_similarity = 0
    else:
        # calculate the similarity of red and green -> if very similar (orange tone) -> most likely background
        red_green_similarity = 1 - (abs(red - green) / (red + green))

    background = red_green_similarity * (1 - gray)
    # very green values with little red will also be detected as background
    if green == max([background, red, green, blue]):
        background = green
        
    road = gray
    building = 1 - background - road

    # set background the similarity of red and green; set road the percentage of gray; set building to the rest
    map_type = {'background': background, 'road': road, 'building': building}
    color_appearance = {'gray': gray, 'red': red, 'green': green, 'blue': blue}

    return map_type, color_appearance


# returns array gray, r, g, b of 0-100 percentage on how the color looks to be more red, green, blue
# based on the r, g, b with the highest values and their diff to lower values
def get_color_appearance(color_rgb):
    r, g, b = color_rgb

    # LIGHT GRAY calculation

    # gives a good representation of how gray a color is based on the similarity of rgb values
    # get the max difference between red, green and blue value
    max_diff = max(abs(r - g), abs(r - b), abs(g - b))
    # shows how big the max difference is in comparison to the half of the smallest rgb value
    diff_comparison = min([r, g, b]) / 2
    gray = 1 - (max_diff / diff_comparison)
    if gray < 0:
        gray = 0
    # gives a bigger value to light gray than dark gray
    lightness = np.mean([r, g, b]) / 210
    if lightness > 1:
        lightness = 1
    gray *= lightness

    # RED, GREEN, BLUE calculation

    # subtract half max value
    # if lower value less than half of max value -> its too minimal to change color
    # example: rgb (20, 80, 100) -> r=20 < max=100/2=50 -> rgb (0, 30, 50)
    #          red value irrelevant for color appearance
    # example: rgb (50, 60, 200) -> r=50 and g=60 < max=200/2=100 -> rgb (0, 0, 100)
    #          red and green value irrelevant for color appearance
    max_diff = int(max([r, g, b]) / 2)
    r, g, b = [(x - max_diff if x > max_diff else 0) for x in [r, g, b]]

    # subtract min value
    # if the lower two values are similar it just grays the first color and doesn't change it
    # example: rgb (60, 63, 100) -> subtract min r=60 -> rgb (0, 3, 40)
    #          red has no and green little influence on the color appearance
    # example: rgb (40, 90, 200) -> subtract min r=40 -> rgb (0, 50, 160) ->
    #          red has no but green still has influence on the color appearance
    min_value = min([r, g, b])
    r, g, b = [(x - min_value) for x in [r, g, b]]

    # calculates r, g, b influence in the color
    sum_value = r + g + b
    r, g, b = [(x / sum_value) for x in [r, g, b]]

    return gray, r, g, b


if __name__ == "__main__":
    main()
