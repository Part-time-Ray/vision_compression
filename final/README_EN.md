# JPEG Encoder/Decoder

Complete JPEG compression and decompression implementation with Huffman coding.

## Core Features

✅ RGB ↔ YCbCr color space conversion  
✅ Chroma subsampling (4:2:0)  
✅ 8×8 block DCT/IDCT transform  
✅ Quantization/Dequantization (standard JPEG quantization tables)  
✅ Zigzag scanning  
✅ DC coefficient differential encoding  
✅ AC coefficient run-length encoding  
⭐ **Huffman encoding/decoding** (complete implementation)  
✅ Complete encoder and decoder

## Project Structure

```
final/
├── huffman.py       # Huffman coding algorithm implementation
├── jpeg_core.py     # JPEG core functions (DCT, quantization, etc.)
├── jpeg_codec.py    # JPEG encoder and decoder classes
├── demo.py          # Demo and test program
├── lena.png         # Test image
├── README.md        # Documentation (Chinese)
├── README_EN.md     # Documentation (English)
└── output/          # Output directory (auto-generated)
```

## Dependencies

```bash
pip install numpy Pillow
```

## Quick Start

### Test all quality levels

```bash
python demo.py
```

Output: Tests 5 quality levels (10, 30, 50, 70, 90), showing compression ratio and PSNR

### Test specific quality

```bash
python demo.py 50
```

Output: Detailed compression statistics, Huffman coding info, comparison images, and error maps

## Python API Usage

```python
from jpeg_codec import JPEGEncoder, JPEGDecoder
from PIL import Image

# Encoding
encoder = JPEGEncoder(quality=50)
encoded_data = encoder.encode_image('lena.png', verbose=False)
encoder.save_encoded(encoded_data, 'output.pkl', verbose=False)

# Decoding
decoder = JPEGDecoder()
reconstructed = decoder.load_and_decode('output.pkl')
Image.fromarray(reconstructed).save('reconstructed.png')
```

## Output Files

After running, the following files will be generated in the `output/` directory:

- `lena_qXX.pkl` - Compressed data
- `lena_recon_qXX.png` - Reconstructed image
- `comparison_qXX.png` - Original vs reconstructed comparison (single quality only)
- `error_map_qXX.png` - Error heat map (single quality only)

## JPEG Encoding Pipeline

1. RGB → YCbCr color space conversion
2. Chroma subsampling (4:2:0)
3. 8×8 block DCT transform
4. Quantization (standard JPEG tables, scaled by quality)
5. Zigzag scanning
6. DC coefficient differential encoding
7. AC coefficient run-length encoding
8. **Huffman encoding** (dynamically built coding tables)

Decoding is the complete reverse process of encoding.

## Performance Results

Test results on Lena 512×512:

| Quality | Size | Ratio | PSNR | Assessment |
|---------|------|-------|------|------------|
| 10  | 15 KB  | 50:1 | 27.3 dB | Fair |
| 30  | 30 KB  | 26:1 | 30.7 dB | Good |
| 50  | 41 KB  | 19:1 | 31.8 dB | Good |
| 70  | 58 KB  | 13:1 | 32.7 dB | Good |
| 90  | 116 KB | 7:1  | 34.2 dB | Good |

Original size: 768 KB

## Technical References

- JPEG Standard (ITU-T T.81 | ISO/IEC 10918-1)
- DCT implementation referenced from lab2
- Quantization tables referenced from lab4
- Color space conversion referenced from lab1

---

NYCU Vision Compression Project
