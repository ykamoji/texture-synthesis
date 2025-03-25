import numpy as np
import time
from skimage import io, color
import matplotlib.pyplot as plt
import random
from scipy.ndimage import binary_dilation
from scipy.signal import convolve2d

images = []
paths = ["D20.png", "Texture2.bmp","english.jpg"]
# paths = ["english.jpg"]
display = True
EIGHT_CONNECTED_NEIGHBOR_KERNEL = np.array([[1., 1., 1.], [1., 0., 1.], [1., 1., 1.]], dtype=np.float64)

SIGMA_COEFF = 6.4
ERROR_THRESHOLD = 0.1

for path in paths:
    images.append((path, io.imread(f'../data/texture/{path}')))

def synthRandomPatch(im, tileSize, numTiles, outSize):
    im = im[:, :, 0:3]
    height, width, _ = im.shape
    totalTiles = numTiles ** 2
    systhesisImage = np.zeros((outSize, outSize, 3))

    pos_x, pos_y = 0, 0
    for _ in range(totalTiles):

        x = random.randint(0, height - tileSize)
        y = random.randint(0, width - tileSize)
        tile = im[x:x + tileSize, y:y + tileSize]

        systhesisImage[pos_x * tileSize:(pos_x + 1) * tileSize, pos_y * tileSize:(pos_y + 1) * tileSize] = tile
        pos_y += 1
        if pos_y >= numTiles:
            pos_x += 1
            pos_y = 0

    return systhesisImage / 255.0


def gauss2D(shape, sigma):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0: h /= sumh
    return h


def reshape_image(image, window_height, window_width, im_height, im_width, strid_x=0, strid_y=0):
    shape = ((im_height - window_height + 1), (im_width - window_width + 1), window_height, window_width)
    strides = (strid_x, strid_y, image.strides[0], image.strides[1])
    stride_image = np.lib.stride_tricks.as_strided(image, shape=shape, strides=strides)

    return stride_image.reshape(-1, window_height, window_width)


def compute_ssd(image_gray, window, mask):
    window_height, window_width = window.shape
    im_height, im_width = image_gray.shape

    valid_image = reshape_image(image_gray, window_height, window_width, im_height, im_width, image_gray.strides[0],
                                image_gray.strides[1])
    valid_window = reshape_image(window, window_height, window_width, im_height, im_width)

    valid_mask = reshape_image(mask, window_height, window_width, im_height, im_width)

    sigma = window_height / SIGMA_COEFF
    kernel = gauss2D((window_height, window_height), sigma=sigma)
    kernel_2d = kernel * kernel.T
    valid_kernel = reshape_image(kernel_2d, window_height, window_width, im_height, im_width)

    squared_differences = ((valid_image - valid_window) ** 2) * valid_kernel * valid_mask
    ssd = np.sum(squared_differences, axis=(1, 2))
    ssd = ssd.reshape(im_height - window_height + 1, im_width - window_width + 1)
    total_ssd = np.sum(mask * kernel_2d)
    normalized_ssd = ssd / total_ssd

    return normalized_ssd


def choose_pixel(ssd):
    min_ssd = np.min(ssd)
    min_threshold = min_ssd * (1. + ERROR_THRESHOLD)
    bestMatches = np.where(ssd <= min_threshold)

    indices_count = bestMatches[0].shape[0]
    weights = np.ones(indices_count) / float(indices_count)
    random_index = np.random.choice(np.arange(indices_count), size=1, p=weights)
    selected_index = (bestMatches[0][random_index], bestMatches[1][random_index])

    return selected_index


def check_masking_completed(total, mask):
    filled_pixels = np.count_nonzero(mask)
    num_incomplete = total - filled_pixels

    percent_completed = filled_pixels * 100 / total
    print(f"{percent_completed:.3f} %", end='\r', flush=True)

    return num_incomplete > 0


def synthEfrosLeung(im, winsize, outSize):
    im = im[:, :, 0:3]

    image_gray = color.rgb2gray(im)

    window = np.zeros((outSize, outSize))

    outputImage = np.zeros((outSize, outSize, 3))

    mask = np.zeros((outSize, outSize))

    sh, sw = im.shape[:2]
    random_x = np.random.randint(sh - 3 + 1)
    random_y = np.random.randint(sw - 3 + 1)

    seed = image_gray[random_x:random_x + 3, random_y:random_y + 3]

    middle_x, middle_y = (outSize // 2) - 1, (outSize // 2) - 1
    window[middle_x:middle_x + 3, middle_y:middle_y + 3] = seed
    mask[middle_x:middle_x + 3, middle_y:middle_y + 3] = 1
    outputImage[middle_x:middle_x + 3, middle_y:middle_y + 3] = im[random_x:random_x + 3, random_y:random_y + 3]

    pad = winsize // 2
    extended_mask = np.pad(mask, pad, 'constant', constant_values=0)
    extended_window = np.pad(window, pad, 'constant', constant_values=0)

    window = extended_window[pad:-pad, pad:-pad]
    mask = extended_mask[pad:-pad, pad:-pad]

    total = mask.shape[:2][0] * mask.shape[:2][1]
    while check_masking_completed(total, mask):

        kernel = np.ones((3, 3))
        pixelList = binary_dilation(mask, kernel) - mask
        pixelList = np.nonzero(pixelList)

        pixel_count = pixelList[0].shape[0]
        permuted_indices = np.random.permutation(np.arange(pixel_count))
        permuted_neighbors = (pixelList[0][permuted_indices], pixelList[1][permuted_indices])
        neighbor = convolve2d(mask, EIGHT_CONNECTED_NEIGHBOR_KERNEL, mode='same')
        permuted_neighbor = neighbor[permuted_neighbors]

        sorted_order = np.argsort(permuted_neighbor)[::-1]
        pixelList = (permuted_neighbors[0][sorted_order], permuted_neighbors[1][sorted_order])

        for ch, cw in zip(pixelList[0], pixelList[1]):

            mask_cut = extended_mask[ch:ch + winsize, cw:cw + winsize]
            window_cut = extended_window[ch:ch + winsize, cw:cw + winsize]

            min_ssd = compute_ssd(image_gray, window_cut, mask_cut)

            index_pixel = choose_pixel(min_ssd)
            index_pixel = (index_pixel[0] + winsize // 2, index_pixel[1] + winsize // 2)

            window[ch, cw] = image_gray[index_pixel].item()
            mask[ch, cw] = 1
            outputImage[ch, cw] = im[index_pixel[0], index_pixel[1]]

    return outputImage / 255.0


def evalSynthRandomPatch():
    # Random patches
    tileSizes = [5, 15, 20, 30, 40]  # specify block sizes
    numTiles = 5
    dashes = '-' * 70
    print(dashes)
    print("Random Synthesis Image \t\t Tile Size \t\t Duration (ms)")
    print(dashes)
    for image, im in images:
        for tileSize in tileSizes:
            outSize = numTiles * tileSize  # calculate output image size
            # implement the following, save the random-patch output and record run-times
            start = time.time()
            im_patch = synthRandomPatch(im.copy(), tileSize, numTiles, outSize)
            imageName = image.ljust(12, ' ')
            print(
                f"{imageName} " + "\t" * 3 + f" {tileSize}x{tileSize} " + "\t" * 3 +
                f" {(time.time() - start) * 1000:.5f}")
            if display:
                plt.imshow(im_patch)
                plt.title(f'{tileSize}x{tileSize} tile size')
                plt.savefig(f"../output/texture/random/{image.split('.')[-2]}_{tileSize}x{tileSize}")
                plt.show()
    print(dashes)


def evalSynthEfrosLeung():
    # Non-parametric Texture Synthesis using Efros & Leung algorithm
    winsizes = [3, 5, 7, 11, 15]  # specify window size (5, 7, 11, 15)
    outSize = 70  # specify size of the output image to be synthesized (square for simplicity)
    # implement the following, save the synthesized image and record the run-times
    dashes = '-' * 110
    print(dashes)
    print("Non-parametric Synth Image \t\t Window Size  \t\t Image size \t\t Duration (sec)")
    print(dashes)
    for image, im in images:
        for winsize in winsizes:
            start = time.time()
            im_synth = synthEfrosLeung(im, winsize, outSize)
            imageName = image.ljust(12, ' ')
            print(
                f"{imageName} " + "\t" * 4 + f" {str(winsize).ljust(3, ' ')} " + "\t" * 3 +
                f" {outSize} x {outSize}" + "\t" * 2 + f" {(time.time() - start):.3f}")
            if display:
                plt.imshow(im_synth)
                plt.title(f"{winsize} window size, {outSize}x{outSize} output size")
                plt.savefig(f"../output/texture/non_parametric/{image.split('.')[-2]}_{winsize}_{outSize}x{outSize}")
                plt.show()
    print(dashes)


if __name__ == '__main__':
    evalSynthRandomPatch()
    evalSynthEfrosLeung()
