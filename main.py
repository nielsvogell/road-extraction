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
    image_name = 'test01'
    img_rgb = cv2.cvtColor(cv2.imread(''.join(['data/', image_name, '.png'])), cv2.COLOR_RGB2BGR)

    cluster_img, cluster_labels, cluster_colors = segment(img_rgb)
    roads = extract_roads(cluster_img, cluster_labels, n_largest=7)
    buildings = extract_buildings(cluster_img, cluster_labels, cluster_colors=cluster_colors, output_type='mask')
    
    img_out = img_rgb.copy()
    img_out[np.where(buildings > 0)] = np.array([250, 120, 120], dtype=np.uint8)
    img_out[np.where(roads > 0)] = np.array([50, 50, 250], dtype=np.uint8)

    grid = ImageGrid(plt.figure(1, figsize=(10, 10)), 111, nrows_ncols=(2, 4), axes_pad=0.1)
    fig = plt.figure(figsize=(10, 10), tight_layout=True)

    gs = fig.add_gridspec(4, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_rgb)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(cluster_img)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(buildings, cmap='gray')
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(roads, cmap='gray')
    ax5 = fig.add_subplot(gs[2:, :])
    ax5.imshow(img_out)

    plt.savefig(''.join(['out/', image_name, '_out.png']), dpi=300)
    
    print('Done')
    
    
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
    print('Done')

