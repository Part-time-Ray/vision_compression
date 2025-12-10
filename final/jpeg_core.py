"""
JPEG Encoder Implementation
Includes: RGB to YCbCr conversion, chroma subsampling, DCT, quantization, 
          zigzag scanning, DC differential encoding, AC run-length encoding
"""
import numpy as np
from PIL import Image


# Standard JPEG quantization tables
JPEG_QUANTIZATION_TABLE_LUMINANCE = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float32)

JPEG_QUANTIZATION_TABLE_CHROMINANCE = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
], dtype=np.float32)


def generate_zigzag_order(n=8):
    """Generate zigzag scan order for 8x8 block"""
    zigzag_order = []
    for d in range(2 * n - 1):
        start_r = max(0, d - (n - 1))
        end_r = min(d, n - 1) + 1
        positions = [(r, d - r) for r in range(start_r, end_r)]
        if d % 2 == 0:
            positions = positions[::-1]
        zigzag_order.extend(positions)
    return zigzag_order


ZIGZAG_ORDER = generate_zigzag_order()


def rgb_to_ycbcr(image):
    """
    Convert RGB image to YCbCr color space
    Args:
        image: numpy array of shape (H, W, 3) with RGB values [0, 255]
    Returns:
        Y, Cb, Cr: three numpy arrays of shape (H, W)
    """
    r = image[:, :, 0].astype(np.float32)
    g = image[:, :, 1].astype(np.float32)
    b = image[:, :, 2].astype(np.float32)
    
    # JPEG standard conversion
    Y = 0.299 * r + 0.587 * g + 0.114 * b
    Cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128
    Cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128
    
    return Y, Cb, Cr


def ycbcr_to_rgb(Y, Cb, Cr):
    """
    Convert YCbCr back to RGB
    Args:
        Y, Cb, Cr: numpy arrays of shape (H, W)
    Returns:
        RGB image: numpy array of shape (H, W, 3)
    """
    Cb = Cb - 128
    Cr = Cr - 128
    
    r = Y + 1.402 * Cr
    g = Y - 0.344136 * Cb - 0.714136 * Cr
    b = Y + 1.772 * Cb
    
    rgb = np.stack([r, g, b], axis=2)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    
    return rgb


def downsample_chroma(channel):
    """
    Downsample chroma channel by 2 (4:2:0 subsampling)
    Args:
        channel: numpy array of shape (H, W)
    Returns:
        downsampled channel: numpy array of shape (H//2, W//2)
    """
    h, w = channel.shape
    # Average 2x2 blocks
    downsampled = np.zeros((h // 2, w // 2), dtype=np.float32)
    for i in range(0, h, 2):
        for j in range(0, w, 2):
            downsampled[i // 2, j // 2] = np.mean(channel[i:i+2, j:j+2])
    return downsampled


def upsample_chroma(channel, target_shape):
    """
    Upsample chroma channel by 2 (reverse 4:2:0 subsampling)
    Args:
        channel: numpy array of shape (H//2, W//2)
        target_shape: tuple (H, W)
    Returns:
        upsampled channel: numpy array of shape (H, W)
    """
    h, w = target_shape
    upsampled = np.zeros((h, w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            upsampled[i, j] = channel[i // 2, j // 2]
    return upsampled


def dct_2d(block):
    """
    Apply 2D DCT to 8x8 block using separable 1D DCT
    Args:
        block: numpy array of shape (8, 8)
    Returns:
        dct_coeffs: numpy array of shape (8, 8)
    """
    def dct_1d_vector(vec):
        N = len(vec)
        result = np.zeros(N, dtype=np.float32)
        for u in range(N):
            cu = 1 / np.sqrt(2) if u == 0 else 1
            result[u] = np.sqrt(2 / N) * cu * np.sum(
                [vec[x] * np.cos((2 * x + 1) * u * np.pi / (2 * N)) for x in range(N)]
            )
        return result
    
    # Apply DCT to rows
    coeffs = np.zeros_like(block, dtype=np.float32)
    for i in range(8):
        coeffs[i, :] = dct_1d_vector(block[i, :])
    
    # Apply DCT to columns
    for j in range(8):
        coeffs[:, j] = dct_1d_vector(coeffs[:, j])
    
    return coeffs


def idct_2d(coeffs):
    """
    Apply 2D inverse DCT to 8x8 block using separable 1D IDCT
    Args:
        coeffs: numpy array of shape (8, 8)
    Returns:
        block: numpy array of shape (8, 8)
    """
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
    
    # Apply IDCT to columns first
    block = np.zeros_like(coeffs, dtype=np.float32)
    for j in range(8):
        block[:, j] = idct_1d_vector(coeffs[:, j])
    
    # Apply IDCT to rows
    for i in range(8):
        block[i, :] = idct_1d_vector(block[i, :])
    
    return block


def quantize(dct_coeffs, q_table):
    """
    Quantize DCT coefficients
    Args:
        dct_coeffs: numpy array of shape (8, 8)
        q_table: quantization table of shape (8, 8)
    Returns:
        quantized: numpy array of shape (8, 8) with integer values
    """
    return np.round(dct_coeffs / q_table).astype(np.int32)


def dequantize(quantized, q_table):
    """
    Dequantize coefficients
    Args:
        quantized: numpy array of shape (8, 8) with integer values
        q_table: quantization table of shape (8, 8)
    Returns:
        dct_coeffs: numpy array of shape (8, 8)
    """
    return (quantized * q_table).astype(np.float32)


def zigzag_scan(block):
    """
    Scan 8x8 block in zigzag order
    Args:
        block: numpy array of shape (8, 8)
    Returns:
        sequence: list of 64 values in zigzag order
    """
    return [block[r, c] for r, c in ZIGZAG_ORDER]


def zigzag_descan(sequence):
    """
    Convert zigzag sequence back to 8x8 block
    Args:
        sequence: list of 64 values in zigzag order
    Returns:
        block: numpy array of shape (8, 8)
    """
    block = np.zeros((8, 8), dtype=np.int32)
    for i, (r, c) in enumerate(ZIGZAG_ORDER):
        if i < len(sequence):
            block[r, c] = sequence[i]
    return block


def encode_dc_coefficient(dc_value, prev_dc_value):
    """
    Encode DC coefficient using differential encoding
    Args:
        dc_value: current DC coefficient
        prev_dc_value: previous DC coefficient
    Returns:
        diff: difference value
        category: magnitude category (0-11)
    """
    diff = dc_value - prev_dc_value
    
    # Determine category (number of bits needed)
    if diff == 0:
        return 0, 0
    
    abs_diff = abs(diff)
    category = int(np.floor(np.log2(abs_diff))) + 1
    
    # Return difference and category
    return diff, category


def encode_ac_coefficients(ac_sequence):
    """
    Encode AC coefficients using run-length encoding
    Args:
        ac_sequence: list of 63 AC coefficients
    Returns:
        encoded: list of (run_length, value) tuples
    """
    encoded = []
    run_length = 0
    
    for value in ac_sequence:
        if value == 0:
            run_length += 1
        else:
            # Handle runs longer than 15
            while run_length > 15:
                encoded.append((15, 0))  # ZRL (Zero Run Length)
                run_length -= 16
            
            encoded.append((run_length, value))
            run_length = 0
    
    # End of Block marker
    if run_length > 0 or not encoded:
        encoded.append((0, 0))  # EOB
    
    return encoded


def decode_ac_coefficients(encoded):
    """
    Decode AC coefficients from run-length encoding
    Args:
        encoded: list of (run_length, value) tuples
    Returns:
        ac_sequence: list of 63 AC coefficients
    """
    ac_sequence = []
    
    for run_length, value in encoded:
        # End of Block
        if run_length == 0 and value == 0:
            break
        
        # Add zeros
        ac_sequence.extend([0] * run_length)
        # Add value
        if not (run_length == 15 and value == 0):  # Skip ZRL marker value
            ac_sequence.append(value)
    
    # Pad to 63 coefficients
    while len(ac_sequence) < 63:
        ac_sequence.append(0)
    
    return ac_sequence[:63]


def scale_quantization_table(base_table, quality):
    """
    Scale quantization table based on quality factor
    Args:
        base_table: base quantization table
        quality: quality factor (1-100), higher means better quality
    Returns:
        scaled_table: scaled quantization table
    """
    if quality < 1:
        quality = 1
    elif quality > 100:
        quality = 100
    
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality
    
    scaled_table = np.floor((base_table * scale + 50) / 100)
    scaled_table[scaled_table < 1] = 1
    
    return scaled_table.astype(np.float32)
