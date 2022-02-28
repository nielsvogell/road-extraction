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
from mpl_toolkits.axes_grid1 import ImageGrid
from importlib import reload


def main():
    print('Type the name of the input file.\n data/...')
    file_name = input()
    if file_name == '':
        file_name = 'test01.png'
    image_name, extension = file_name.split('.')
    
    # Read image as RGB
    img_rgb = cv2.cvtColor(cv2.imread(''.join(['data/', file_name])), cv2.COLOR_RGB2BGR)

    # Get the segmented image from the segment module
    cluster_img, cluster_labels, cluster_colors = segment(img_rgb)
    
    # Create colored image of segmentation result for later plotting
    (h, w) = cluster_img.shape
    color_eval_img = np.zeros((h, w, 3), dtype=np.uint8)
    if cluster_colors is not None:
        for lbl, color in cluster_colors.items():
            color_eval_img[np.where(cluster_img == lbl)] = color.astype(np.uint8)
    
    # Extract the roads using the road_extraction module
    roads = extract_roads(cluster_img, cluster_labels, n_largest=7)
    
    # Extract the buildings using the building extraction module
    buildings = extract_buildings(cluster_img, cluster_labels, cluster_colors=cluster_colors, output_type='mask')
    
    # Create a final image from the original image, with buildings and roads drawn on top
    img_out = img_rgb.copy()
    img_out[np.where(buildings > 0)] = np.array([250, 120, 120], dtype=np.uint8)
    img_out[np.where(roads > 0)] = np.array([50, 50, 250], dtype=np.uint8)
    
    # Output
    # - Original image
    # - segmented image
    # - detected roads
    # - detected buildings
    # - overlay of buildings and roads on input image
    fig = plt.figure(figsize=(10, 10), tight_layout=True)

    gs = fig.add_gridspec(4, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    ax1.set_title('original')
    ax1.imshow(img_rgb)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('segmented image')
    ax2.axis('off')
    ax2.imshow(color_eval_img)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title('detected buildings')
    ax3.axis('off')
    ax3.imshow(buildings, cmap='gray')
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_title('road network')
    ax4.axis('off')
    ax4.imshow(roads, cmap='gray')
    ax5 = fig.add_subplot(gs[2:, :])
    ax5.set_title('joined output')
    ax5.axis('off')
    ax5.imshow(img_out)

    plt.savefig(''.join(['out/', image_name, '_out.png']), dpi=300)
    print('Saved output file to', ''.join(['out/', image_name, '_out.png']), '\n')
    
    
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
    print('Done')

