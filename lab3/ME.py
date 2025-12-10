import numpy as np
import time
from PIL import Image
    
frame1 = Image.open('one_gray.png').convert('L')
frame2 = Image.open('two_gray.png').convert('L')

frame1 = np.array(frame1)
frame2 = np.array(frame2)

def full_search_block_matching(img1, img2, block_size=8, search_range=8):
    h, w = img1.shape
    mv_x = np.zeros((h // block_size, w // block_size), dtype=np.int32)
    mv_y = np.zeros((h // block_size, w // block_size), dtype=np.int32)
    reconstructed = np.zeros_like(img1)
    
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            best_sad = 1e9
            best_dx, best_dy = 0, 0
            ref_block = img1[i:i+block_size, j:j+block_size]
            
            for dy in range(-search_range, search_range + 1):
                for dx in range(-search_range, search_range + 1):
                    y = i + dy
                    x = j + dx
                    if y < 0 or y + block_size > h or x < 0 or x + block_size > w:
                        continue
                    cand_block = img2[y:y+block_size, x:x+block_size]
                    sad = np.sum(np.abs(ref_block - cand_block))
                    if sad < best_sad:
                        best_sad = sad
                        best_dx, best_dy = dx, dy
            
            mv_x[i//block_size, j//block_size] = best_dx
            mv_y[i//block_size, j//block_size] = best_dy
            reconstructed[i:i+block_size, j:j+block_size] = img2[i+best_dy:i+best_dy+block_size, j+best_dx:j+best_dx+block_size]
    
    residual = img1.astype(np.int16) - reconstructed.astype(np.int16)
    return reconstructed, residual, mv_x, mv_y

def three_step_search(img1, img2, block_size=8, search_range=8):
    h, w = img1.shape
    mv_x = np.zeros((h // block_size, w // block_size), dtype=np.int32)
    mv_y = np.zeros((h // block_size, w // block_size), dtype=np.int32)
    reconstructed = np.zeros_like(img1)

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            ref_block = img1[i:i+block_size, j:j+block_size]
            cx, cy = j, i
            best_dx, best_dy = 0, 0
            step = search_range // 2
            while step >= 1:
                best_sad = 1e9
                for dy in [-step, 0, step]:
                    for dx in [-step, 0, step]:
                        x = cx + dx
                        y = cy + dy
                        if y < 0 or y + block_size > h or x < 0 or x + block_size > w:
                            continue
                        cand_block = img2[y:y+block_size, x:x+block_size]
                        sad = np.sum(np.abs(ref_block - cand_block))
                        if sad < best_sad:
                            best_sad = sad
                            best_dx, best_dy = x - j, y - i
                            best_x, best_y = x, y
                cx, cy = best_x, best_y
                step //= 2

            mv_x[i//block_size, j//block_size] = best_dx
            mv_y[i//block_size, j//block_size] = best_dy
            reconstructed[i:i+block_size, j:j+block_size] = img2[i+best_dy:i+best_dy+block_size, j+best_dx:j+best_dx+block_size]

    residual = img1.astype(np.int16) - reconstructed.astype(np.int16)
    return reconstructed, residual, mv_x, mv_y

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(255 / np.sqrt(mse))

for sr in [8, 16, 32]:
    start = time.time()
    rec, res, mvx, mvy = full_search_block_matching(frame2, frame1, 8, sr)
    runtime = time.time() - start
    print(f"Full Search range ±{sr}: PSNR={psnr(frame2, rec):.2f}, Time={runtime:.2f}s")
    # if sr == 8:
    Image.fromarray(rec).save(f'reconstructed_full_{sr}.png')
    res_shifted = np.abs(res).clip(0, 255).astype(np.uint8)
    Image.fromarray(res_shifted).save(f'residual_full_{sr}.png')

for sr in [8, 16, 32]:
    start = time.time()
    rec_tss, res_tss, mvx_tss, mvy_tss = three_step_search(frame2, frame1, 8, sr)
    runtime_tss = time.time() - start
    print(f"Three-Step Search range ±{sr}: PSNR={psnr(frame2, rec_tss):.2f}, Time={runtime_tss:.2f}s")
    Image.fromarray(rec_tss).save(f'reconstructed_three_step_{sr}.png')
    res_tss_shifted = np.abs(res_tss).clip(0, 255).astype(np.uint8)
    Image.fromarray(res_tss_shifted).save(f'residual_three_step_{sr}.png')