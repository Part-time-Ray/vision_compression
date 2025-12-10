import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import os
import pickle

q_table1 = np.array([
    [10, 7, 6, 10, 14, 24, 31, 37],
    [7, 7, 8, 11, 16, 35, 36, 33],
    [8, 8, 10, 14, 24, 34, 41, 34],
    [8, 10, 13, 17, 31, 52, 48, 37],
    [11, 13, 22, 34, 41, 65, 62, 46],
    [14, 21, 33, 38, 49, 62, 68, 55],
    [29, 38, 47, 52, 62, 73, 72, 61],
    [43, 55, 57, 59, 67, 60, 62, 59]
])

q_table2 = np.array([
    [10, 11, 14, 28, 59, 59, 59, 59],
    [11, 13, 16, 40, 59, 59, 59, 59],
    [14, 16, 34, 59, 59, 59, 59, 59],
    [28, 40, 59, 59, 59, 59, 59, 59],
    [59, 59, 59, 59, 59, 59, 59, 59],
    [59, 59, 59, 59, 59, 59, 59, 59],
    [59, 59, 59, 59, 59, 59, 59, 59],
    [59, 59, 59, 59, 59, 59, 59, 59]
])

def generate_zigzag_order(n=8):
    zigzag_order = []
    for d in range(2 * n - 1):
        start_r = max(0, d - (n - 1))
        end_r = min(d, n - 1) + 1
        positions = [(r, d - r) for r in range(start_r, end_r)]
        if d % 2 == 0:
            positions = positions[::-1]
        zigzag_order.extend(positions)
    return zigzag_order

zigzag_order = generate_zigzag_order()

def rle_encode(seq):
    encoded = []
    run = 0
    for val in seq:
        if val == 0:
            run += 1
        else:
            encoded.append((run, val))
            run = 0
    return encoded

def rle_decode(encoded, size=64):
    seq = []
    for run, val in encoded:
        seq.extend([0] * run)
        seq.append(val)
    seq.extend([0] * (size - len(seq)))
    return seq

def dct_2d(block):
    def dct_1d_vector(vec):
        N = len(vec)
        result = np.zeros(N, dtype=np.float32)
        for u in range(N):
            cu = 1 / np.sqrt(2) if u == 0 else 1
            result[u] = np.sqrt(2 / N) * cu * np.sum([vec[x] * np.cos((2 * x + 1) * u * np.pi / (2 * N)) for x in range(N)])
        return result
    
    coff = np.zeros_like(block, dtype=np.float32)
    for i in range(8):
        coff[i, :] = dct_1d_vector(block[i, :])
    for j in range(8):
        coff[:, j] = dct_1d_vector(coff[:, j])
    return coff

def idct_2d(coff):
    def idct_1d_vector(vec):
        N = len(vec)
        result = np.zeros(N, dtype=np.float32)
        for x in range(N):
            sum_val = 0
            for u in range(N):
                cu = 1 / np.sqrt(2) if u == 0 else 1
                sum_val += cu * vec[u] * np.cos((2 * x + 1) * u * np.pi / (2 * N))
            result[x] = np.sqrt(2 / N) * sum_val
        return result
    
    block_rec = np.zeros_like(coff, dtype=np.float32)
    for j in range(8):
        block_rec[:, j] = idct_1d_vector(coff[:, j])
    for i in range(8):
        block_rec[i, :] = idct_1d_vector(block_rec[i, :])
    return block_rec

def process_image(q_table, table_name):
    image = Image.open('lena.png').convert('L')
    image = np.array(image, dtype=np.float32)
    plt.imsave(f'original_{table_name}.png', image / 255, cmap='gray')
    
    h, w = image.shape
    num_blocks_h = h // 8
    num_blocks_w = w // 8
    
    encoded_all = []
    
    tq = tqdm(total=num_blocks_h * num_blocks_w, desc=f'Encoding with {table_name}')
    for bi in range(num_blocks_h):
        for bj in range(num_blocks_w):
            block = image[bi*8:(bi+1)*8, bj*8:(bj+1)*8] - 128
            coeff = dct_2d(block)
            quant = np.round(coeff / q_table).astype(int)

            seq = [quant[r, c] for r, c in zigzag_order]
            enc = rle_encode(seq)
            encoded_all.append(enc)
            tq.update(1)
    tq.close()
    
    encoded_file = f'encoded_{table_name}.pkl'
    with open(encoded_file, 'wb') as f:
        pickle.dump(encoded_all, f)
    size = os.path.getsize(encoded_file)
    
    recovered = np.zeros_like(image)
    block_idx = 0
    tq = tqdm(total=num_blocks_h * num_blocks_w, desc=f'Decoding with {table_name}')
    for bi in range(num_blocks_h):
        for bj in range(num_blocks_w):
            enc = encoded_all[block_idx]
            seq = rle_decode(enc)
            
            quant = np.zeros((8, 8), dtype=int)
            for k, (r, c) in enumerate(zigzag_order):
                quant[r, c] = seq[k]
            coeff = quant * q_table.astype(np.float32)
            
            block_rec = idct_2d(coeff) + 128
            block_rec = np.clip(block_rec, 0, 255)
            recovered[bi*8:(bi+1)*8, bj*8:(bj+1)*8] = block_rec
            block_idx += 1
            tq.update(1)
    tq.close()
    
    plt.imsave(f'reconstructed_{table_name}.png', recovered / 255, cmap='gray')
    
    return size, image, recovered

size1, orig1, rec1 = process_image(q_table1, '1')
size2, orig2, rec2 = process_image(q_table2, '2')

def psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

print(f'Comparison: Table 1 size = {size1} bytes, PSNR = {psnr(orig1, rec1):.2f} dB')
print(f'Comparison: Table 2 size = {size2} bytes, PSNR = {psnr(orig2, rec2):.2f} dB')