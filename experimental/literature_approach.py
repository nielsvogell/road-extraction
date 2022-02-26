import numpy as np
import cv2
import scipy.signal as sps
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


plt.ion()
## Image segmentation
img = cv2.cvtColor(cv2.imread('data/exp.png'), cv2.COLOR_BGR2RGB)

# contrast strecthing
sorted_intensities = np.sort(img.reshape(-1, 3), axis=0)
min_intensity = sorted_intensities[int(0.05*sorted_intensities.shape[0]), :].astype(np.float64)
max_intensity = sorted_intensities[int(0.95*sorted_intensities.shape[0]), :].astype(np.float64)

img_normalized = np.zeros_like(img)
c = 0  # for testing
for c in range(3):
    img_stretched = img[:, :, c].astype(np.float64)
    img_stretched[np.where(img_stretched < min_intensity[c])] = min_intensity[c]
    img_stretched[np.where(img_stretched > max_intensity[c])] = max_intensity[c]
    img_stretched = ((img_stretched - min_intensity[c]) / (max_intensity[c] - min_intensity[c]) * 255)
    plt.clf()
    plt.hist(img_stretched.ravel(), bins=256)
    plt.hist(img[:,:,2].ravel(), bins=256)
    plt.clf()
    plt.imshow(img_stretched, cmap='gray')
    
    # standard deviation
    img_stretched_mean = cv2.blur(img_stretched.astype(np.float64), (3, 3))
    img_stretched_sq_mean = cv2.blur(np.power(img_stretched.astype(np.float64), 2), (3, 3))
    non_negatives = (img_stretched_sq_mean - np.power(img_stretched_mean, 2)) > 0
    std_deviation = np.zeros_like(img_stretched)
    std_deviation[non_negatives] = np.sqrt(img_stretched_sq_mean[non_negatives] - np.power(img_stretched_mean[non_negatives], 2))
    # plt.clf()
    # plt.imshow(std_deviation, cmap='gray')
    
    plt.clf()
    plt.hist((img_stretched - img_stretched_mean).ravel(), bins=256)
    plt.hist(std_deviation.ravel(), bins=256)
    z_score = np.divide(img_stretched - img_stretched_mean, std_deviation,
                        out=np.zeros_like(img_stretched, dtype=np.float64),
                        where=std_deviation > 0)
    plt.clf()
    plt.hist(z_score.ravel(), bins=256)
    img_normalized[:, :, c] = np.round((z_score - np.min(z_score))/(np.max(z_score) - np.min(z_score))*255).astype(np.uint8)
plt.subplot(2, 2, 1)
plt.imshow(img_normalized)
plt.subplot(2, 2, 2)
plt.imshow(img_normalized[:, :, 0], cmap='gray')
plt.subplot(2, 2, 3)
plt.imshow(img_normalized[:, :, 1], cmap='gray')
plt.subplot(2, 2, 4)
plt.imshow(img_normalized[:, :, 2], cmap='gray')

# Get the homogeneity histogram
r_hist = cv2.calcHist(img_normalized, [0], None, [256], (0, 256), accumulate=False)
g_hist = cv2.calcHist(img_normalized, [1], None, [256], (0, 256), accumulate=False)
b_hist = cv2.calcHist(img_normalized, [2], None, [256], (0, 256), accumulate=False)

hist_w = 512
hist_h = 400
cv2.normalize(b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
cv2.normalize(g_hist, g_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
cv2.normalize(r_hist, r_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
bin_w = int(round(hist_w / 256))
histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
for i in range(1, 256):
    cv2.line(histImage, (bin_w * (i-1), hist_h - int(b_hist[i-1])),
             (bin_w * i, hist_h - int(b_hist[i])),
             (255, 0, 0), thickness=2)
    cv2.line(histImage, (bin_w * (i-1), hist_h - int(g_hist[i-1])),
             (bin_w * i, hist_h - int(g_hist[i])),
             (0, 255, 0), thickness=2)
    cv2.line(histImage, ( bin_w*(i-1), hist_h - int(r_hist[i-1])),
             (bin_w * i, hist_h - int(r_hist[i])),
             (0, 0, 255), thickness=2)
plt.imshow(histImage)


r_hist_smooth = cv2.GaussianBlur(r_hist, (1, 15), sigmaX=3.5)
g_hist_smooth = cv2.GaussianBlur(g_hist, (1, 15), sigmaX=3.5)
b_hist_smooth = cv2.GaussianBlur(b_hist, (1, 15), sigmaX=3.5)
b_hist_smooth_norm = cv2.normalize(b_hist_smooth, np.zeros_like(r_hist_smooth), alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
g_hist_smooth_norm = cv2.normalize(g_hist_smooth, np.zeros_like(r_hist_smooth), alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
r_hist_smooth_norm = cv2.normalize(r_hist_smooth, np.zeros_like(r_hist_smooth), alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

histImageSmooth = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
for i in range(1, 256):
    cv2.line(histImageSmooth, (bin_w * (i-1), hist_h - int(hist_h * r_hist_smooth_norm[i-1])),
             (bin_w * i, hist_h - int(hist_h * r_hist_smooth_norm[i])),
             (255, 0, 0), thickness=1)
    cv2.line(histImageSmooth, (bin_w * (i-1), hist_h - int(hist_h * g_hist_smooth_norm[i-1])),
             (bin_w * i, hist_h - int(hist_h * g_hist_smooth_norm[i])),
             (0, 255, 0), thickness=1)
    cv2.line(histImageSmooth, ( bin_w*(i-1), hist_h - int(hist_h * b_hist_smooth_norm[i-1])),
             (bin_w * i, hist_h - int(hist_h * b_hist_smooth_norm[i])),
             (0, 0, 255), thickness=1)
plt.imshow(histImageSmooth)

# Find minima
r_hist_maxfilter = cv2.dilate(r_hist_smooth, np.ones((10, 1)))
g_hist_maxfilter = cv2.dilate(g_hist_smooth, np.ones((10, 1)))
b_hist_maxfilter = cv2.dilate(b_hist_smooth, np.ones((10, 1)))
r_hist_minfilter = cv2.erode(r_hist_smooth, np.ones((10, 1)))
g_hist_minfilter = cv2.erode(g_hist_smooth, np.ones((10, 1)))
b_hist_minfilter = cv2.erode(b_hist_smooth, np.ones((10, 1)))
r_maxima = sps.argrelmax(r_hist_smooth)[0]
r_maxima = r_maxima[np.where(r_hist_smooth[r_maxima] - r_hist_minfilter[r_maxima] > 1)[0]]
r_minima = sps.argrelmin(r_hist_smooth)[0]
r_minima = (r_minima[np.where(r_hist_maxfilter[r_minima] - r_hist_smooth[r_minima] > 1)[0]])
g_minima = sps.argrelmin(g_hist_smooth)[0]
g_minima = (g_minima[np.where(g_hist_maxfilter[g_minima] - g_hist_smooth[g_minima] > 1)[0]])
b_minima = sps.argrelmin(b_hist_smooth)[0]
b_minima = (b_minima[np.where(b_hist_maxfilter[b_minima] - b_hist_smooth[b_minima] > 1)[0]])
plt.clf()
plt.plot(r_hist_smooth)
plt.scatter(r_maxima, r_hist_smooth[r_maxima])
plt.scatter(r_minima, r_hist_smooth[r_minima])

r_mask = np.zeros_like(img[:, :, 0])
b_mask = np.zeros_like(img[:, :, 0])
g_mask = np.zeros_like(img[:, :, 0])
for i in range(len(r_minima) - 1):
    r_mask[np.where(cv2.inRange(img_normalized[:, :, 0], int(r_minima[i]), int(r_minima[i + 1])))] = i + 1
for i in range(len(g_minima) - 1):
    g_mask[np.where(cv2.inRange(img_normalized[:, :, 0], int(g_minima[i]), int(g_minima[i + 1])))] = i + 1
for i in range(len(b_minima) - 1):
    b_mask[np.where(cv2.inRange(img_normalized[:, :, 0], int(b_minima[i]), int(b_minima[i + 1])))] = i + 1

plt.clf()
plt.imshow(r_mask / np.max(r_mask) * 255)

for i in range(len(r_minima) - 1):
    r_mask = np.zeros_like(img[:, :, 0])
    r_mask[np.where(cv2.inRange(img_normalized[:, :, 0], int(r_minima[i]), int(r_minima[i + 1])))] = 1
    plt.imshow(r_mask / np.max(r_mask) * 255)
    plt.clf()

print("done")
## Road network extraction


