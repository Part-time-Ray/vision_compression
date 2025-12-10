"""
JPEG Compression Demo
Usage: python demo.py [quality]
       python demo.py 50        # Test quality 50 only
       python demo.py           # Test all quality levels
"""
import numpy as np
from PIL import Image
import os
import sys
import argparse

from jpeg_codec import JPEGEncoder, JPEGDecoder


def calculate_psnr(original, reconstructed):
    """Calculate PSNR between original and reconstructed images"""
    original = original.astype(np.float32)
    reconstructed = reconstructed.astype(np.float32)
    
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return psnr


def calculate_actual_compressed_size(encoded_data):
    """
    Calculate actual compressed data size (not including Python overhead)
    This estimates the size of the actual JPEG-like bitstream
    """
    total_bits = 0
    
    # Count DC coefficients (using category bits)
    for block in encoded_data['encoded_y']:
        cat = block['dc_category']
        total_bits += cat  # Category value itself
        total_bits += 4  # Category code (simplified)
    
    for block in encoded_data['encoded_cb']:
        cat = block['dc_category']
        total_bits += cat
        total_bits += 4
    
    for block in encoded_data['encoded_cr']:
        cat = block['dc_category']
        total_bits += cat
        total_bits += 4
    
    # Count AC coefficients
    for blocks in [encoded_data['encoded_y'], encoded_data['encoded_cb'], encoded_data['encoded_cr']]:
        for block in blocks:
            for run_length, value in block['ac_encoded']:
                if run_length == 0 and value == 0:
                    total_bits += 4  # EOB
                elif run_length == 15 and value == 0:
                    total_bits += 8  # ZRL
                else:
                    # Run-length (4 bits) + category (4 bits) + value (category bits)
                    if value != 0:
                        abs_value = abs(value)
                        category = int(np.floor(np.log2(abs_value))) + 1 if abs_value > 0 else 0
                        total_bits += 4 + 4 + category
                    else:
                        total_bits += 8
    
    # Add header overhead (estimated)
    header_bits = 1024 * 8  # About 1KB for headers
    
    total_bits += header_bits
    total_bytes = (total_bits + 7) // 8
    
    return total_bytes


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='JPEG Compression Demo')
    parser.add_argument('quality', nargs='?', type=int, default=None,
                        help='Quality level (1-100). If not specified, test all levels.')
    args = parser.parse_args()
    
    # Find lena.png
    image_path = 'lena.png'
    if not os.path.exists(image_path):
        print("Error: lena.png not found in current directory")
        return
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Load original image
    original_image = np.array(Image.open(image_path).convert('RGB'))
    h, w = original_image.shape[:2]
    original_size = h * w * 3
    
    # Determine quality levels to test
    if args.quality is not None:
        if args.quality < 1 or args.quality > 100:
            print("Error: Quality must be between 1 and 100")
            return
        qualities = [args.quality]
        print("=" * 80)
        print(f"JPEG COMPRESSION - Quality {args.quality}")
        print("=" * 80)
    else:
        qualities = [10, 30, 50, 70, 90]
        print("=" * 80)
        print("JPEG COMPRESSION - Testing Multiple Quality Levels")
        print("=" * 80)
    
    print(f"\nOriginal Image: {w}x{h}, {original_size:,} bytes ({original_size/1024:.2f} KB)")
    print()
    # Header
    if len(qualities) > 1:
        print(f"{'Quality':<10} {'Est. Size':<15} {'Ratio':<12} {'PSNR (dB)':<12} {'Assessment':<12}")
        print("-" * 80)
    
    results = []
    for quality in qualities:
        # Encode (suppress verbose output)
        encoder = JPEGEncoder(quality=quality, use_chroma_subsampling=True)
        encoded_data = encoder.encode_image(image_path, verbose=False)
        
        # Calculate estimated compressed size
        estimated_size = calculate_actual_compressed_size(encoded_data)
        
        # Save encoded data
        encoded_path = f'output/lena_q{quality}.pkl'
        encoder.save_encoded(encoded_data, encoded_path, verbose=False)
        
        # Decode (suppress verbose output)
        decoder = JPEGDecoder()
        reconstructed_image = decoder.decode_image(encoded_data, verbose=False)
        
        # Save reconstructed image
        Image.fromarray(reconstructed_image).save(f'output/lena_recon_q{quality}.png')
        
        # Calculate PSNR
        psnr = calculate_psnr(original_image, reconstructed_image)
        
        # Calculate compression ratio
        compression_ratio = original_size / estimated_size
        
        # Determine quality description
        if psnr >= 40:
            quality_desc = "Excellent"
        elif psnr >= 35:
            quality_desc = "Very Good"
        elif psnr >= 30:
            quality_desc = "Good"
        elif psnr >= 25:
            quality_desc = "Fair"
        else:
            quality_desc = "Poor"
        
        results.append({
            'quality': quality,
            'size': estimated_size,
            'ratio': compression_ratio,
            'psnr': psnr,
            'desc': quality_desc
        })
        
        if len(qualities) > 1:
            print(f"{quality:<10} {str(estimated_size)+' B':<15}{compression_ratio:>6.2f}:1     {psnr:>6.2f} dB     {quality_desc:<12}")
    
    if len(qualities) > 1:
        print("-" * 80)
    
    # Detailed output for single quality test
    if len(qualities) == 1:
        result = results[0]
        quality = result['quality']
        
        # Get the encoded data for this quality
        encoder = JPEGEncoder(quality=quality, use_chroma_subsampling=True)
        encoded_data = encoder.encode_image(image_path, verbose=False)
        decoder = JPEGDecoder()
        reconstructed = decoder.decode_image(encoded_data, verbose=False)
        
        print("\n" + "=" * 80)
        print("COMPRESSION RESULTS")
        print("=" * 80)
        print(f"Quality level:        {quality}")
        print(f"Estimated size:       {result['size']:,} bytes ({result['size']/1024:.2f} KB)")
        print(f"Compression ratio:    {result['ratio']:.2f}:1")
        print(f"Space savings:        {100*(1-1/result['ratio']):.2f}%")
        print(f"PSNR:                 {result['psnr']:.2f} dB")
        print(f"Quality assessment:   {result['desc']}")
        
        # Create visual comparison
        comparison = np.hstack([original_image, reconstructed])
        Image.fromarray(comparison).save(f'output/comparison_q{quality}.png')
        
        # Create error map
        error = np.abs(original_image.astype(np.float32) - reconstructed.astype(np.float32))
        error_normalized = (error * 255 / error.max()).astype(np.uint8)
        Image.fromarray(error_normalized).save(f'output/error_map_q{quality}.png')
        
        print("\n" + "=" * 80)
        print("HUFFMAN CODING STATISTICS")
        print("=" * 80)
        print(f"DC Y codes:  {len(encoded_data['huffman_dc_y_table']):>3} symbols")
        print(f"AC Y codes:  {len(encoded_data['huffman_ac_y_table']):>3} symbols")
        print(f"DC C codes:  {len(encoded_data['huffman_dc_c_table']):>3} symbols")
        print(f"AC C codes:  {len(encoded_data['huffman_ac_c_table']):>3} symbols")
        
        print("\nExample Huffman codes (DC Y):")
        for i, (symbol, code) in enumerate(list(encoded_data['huffman_dc_y_table'].items())[:15]):
            print(f"  Symbol {symbol}: {code}")
        
        print("\n" + "=" * 80)
        print("OUTPUT FILES")
        print("=" * 80)
        print(f"Compressed data:      output/lena_q{quality}.pkl")
        print(f"Reconstructed image:  output/lena_recon_q{quality}.png")
        print(f"Comparison image:     output/comparison_q{quality}.png")
        print(f"Error map:            output/error_map_q{quality}.png")
        print("=" * 80)
    else:
        # Summary for multiple quality tests
        print("\n" + "=" * 80)
        print("OUTPUT FILES")
        print("=" * 80)
        print("Compressed data:      output/lena_q*.pkl")
        print("Reconstructed images: output/lena_recon_q*.png")
        print("=" * 80)


if __name__ == '__main__':
    main()
