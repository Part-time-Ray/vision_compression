import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image_size', '-s', type=int, default=128, help='Size of the input image (assumed square)')
parser.add_argument('--mode', '-m', type=str, default='2d', choices=['2d', '1d'])
args = parser.parse_args()

image = Image.open('lena.png').convert('L')
image = image.resize((args.image_size, args.image_size), Image.BICUBIC)
image = np.array(image)

def psnr(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


def dct_2d(image):
    plt.imsave('2D_DCT_original.png', image, cmap='gray')
    coff = np.zeros_like(image, dtype=np.float32)
    tq = tqdm.tqdm(total=image.shape[0] * image.shape[1])
    N = image.shape[0]
    for i in range(N):
        for j in range(N):
            ci = 1 / np.sqrt(2) if i==0 else 1
            cj = 1 / np.sqrt(2) if j==0 else 1
            coff[i, j] = 2 / N * ci * cj * np.sum([image[k, l] * np.cos((2 * k + 1) * i * np.pi / (2 * N)) * np.cos((2 * l + 1) * j * np.pi / (2 * N)) for k in range(N) for l in range(N)])
            tq.update(1)
    tq.close()
    log_coff = np.log(np.abs(coff) + 1)
    plt.imsave('2D_DCT.png', log_coff, cmap='gray')
    return coff

def inv_dct_2d(coff):
    image = np.zeros_like(coff, dtype=np.float32)
    tq = tqdm.tqdm(total=coff.shape[0] * coff.shape[1])
    N = coff.shape[0]
    for k in range(N):
        for l in range(N):
            for u in range(N):
                for v in range(N):
                    cu = 1 / np.sqrt(2) if u==0 else 1
                    cv = 1 / np.sqrt(2) if v==0 else 1
                    image[k, l] += 2 / N * cu * cv * coff[u, v] * np.cos((2 * k + 1) * u * np.pi / (2 * N)) * np.cos((2 * l + 1) * v * np.pi / (2 * N))
            tq.update(1)
    tq.close()
    image = np.clip(image, 0, 255).astype(np.uint8)
    plt.imsave('2D_DCT_reconstructed.png', image, cmap='gray')
    return image

def dct_1d(image):
    plt.imsave('1D_DCT_original.png', image, cmap='gray')
    def dct_1d_vector(vec):
        N = len(vec)
        result = np.zeros(N, dtype=np.float32)
        for u in range(N):
            cu = 1 / np.sqrt(2) if u==0 else 1
            result[u] = np.sqrt(2 / N) * cu * np.sum([vec[x] * np.cos((2 * x + 1) * u * np.pi / (2 * N)) for x in range(N)])
        return result
    coff = np.zeros_like(image, dtype=np.float32)
    tq = tqdm.tqdm(total=image.shape[0] + image.shape[1])
    for i in range(image.shape[0]):
        coff[i, :] = dct_1d_vector(image[i, :])
        tq.update(1)
    for j in range(image.shape[1]):
        coff[:, j] = dct_1d_vector(coff[:, j])
        tq.update(1)
    tq.close()
    log_coff = np.log(np.abs(coff) + 1)
    plt.imsave('1D_DCT.png', log_coff, cmap='gray')
    return coff

def inv_dct_1d(coff):
    def inv_dct_1d_vector(vec):
        N = len(vec)
        result = np.zeros(N, dtype=np.float32)
        for x in range(N):
            sum_val = 0
            for u in range(N):
                cu = 1 / np.sqrt(2) if u==0 else 1
                sum_val += cu * vec[u] * np.cos((2 * x + 1) * u * np.pi / (2 * N))
            result[x] = np.sqrt(2 / N) * sum_val
        return result
    image = np.zeros_like(coff, dtype=np.float32)
    tq = tqdm.tqdm(total=coff.shape[0] + coff.shape[1])
    for j in range(coff.shape[1]):
        image[:, j] = inv_dct_1d_vector(coff[:, j])
        tq.update(1)
    for i in range(coff.shape[0]):
        image[i, :] = inv_dct_1d_vector(image[i, :])
        tq.update(1)
    tq.close()
    image = np.clip(image, 0, 255).astype(np.uint8)
    plt.imsave('1D_DCT_reconstructed.png', image, cmap='gray')
    return image

if args.mode == '2d':
    print('2D-DCT, PSNR:', psnr(image, inv_dct_2d(dct_2d(image))))
elif args.mode == '1d':
    print('1D-DCT, PSNR:', psnr(image, inv_dct_1d(dct_1d(image))))
else:
    raise NotImplementedError