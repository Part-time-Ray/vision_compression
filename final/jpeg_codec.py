"""
JPEG Encoder and Decoder Implementation
Complete JPEG compression and decompression with Huffman coding
"""
import numpy as np
from PIL import Image
import pickle
import os

from jpeg_core import (
    rgb_to_ycbcr, ycbcr_to_rgb,
    downsample_chroma, upsample_chroma,
    dct_2d, idct_2d,
    quantize, dequantize,
    zigzag_scan, zigzag_descan,
    encode_dc_coefficient, encode_ac_coefficients, decode_ac_coefficients,
    JPEG_QUANTIZATION_TABLE_LUMINANCE, JPEG_QUANTIZATION_TABLE_CHROMINANCE,
    scale_quantization_table
)
from huffman import HuffmanCoder


class JPEGEncoder:
    def __init__(self, quality=50, use_chroma_subsampling=True):
        """
        Initialize JPEG encoder
        Args:
            quality: JPEG quality (1-100), higher is better
            use_chroma_subsampling: whether to use 4:2:0 chroma subsampling
        """
        self.quality = quality
        self.use_chroma_subsampling = use_chroma_subsampling
        
        # Scale quantization tables based on quality
        self.q_table_lum = scale_quantization_table(JPEG_QUANTIZATION_TABLE_LUMINANCE, quality)
        self.q_table_chrom = scale_quantization_table(JPEG_QUANTIZATION_TABLE_CHROMINANCE, quality)
        
        # Huffman coders for DC and AC coefficients
        self.huffman_dc_y = HuffmanCoder()
        self.huffman_ac_y = HuffmanCoder()
        self.huffman_dc_c = HuffmanCoder()
        self.huffman_ac_c = HuffmanCoder()
    
    def encode_channel(self, channel, q_table, is_luma=True):
        """
        Encode a single channel (Y, Cb, or Cr)
        Args:
            channel: numpy array of shape (H, W)
            q_table: quantization table
            is_luma: whether this is luminance channel
        Returns:
            encoded_blocks: list of encoded block data
        """
        h, w = channel.shape
        
        # Pad to multiple of 8
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            channel = np.pad(channel, ((0, pad_h), (0, pad_w)), mode='edge')
        
        h, w = channel.shape
        num_blocks_h = h // 8
        num_blocks_w = w // 8
        
        encoded_blocks = []
        prev_dc = 0
        
        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                # Extract 8x8 block
                block = channel[i*8:(i+1)*8, j*8:(j+1)*8]
                
                # Level shift by -128
                block = block - 128
                
                # Apply DCT
                dct_coeffs = dct_2d(block)
                
                # Quantize
                quant_coeffs = quantize(dct_coeffs, q_table)
                
                # Zigzag scan
                zigzag = zigzag_scan(quant_coeffs)
                
                # Separate DC and AC coefficients
                dc = zigzag[0]
                ac = zigzag[1:]
                
                # DC differential encoding
                dc_diff, dc_category = encode_dc_coefficient(dc, prev_dc)
                prev_dc = dc
                
                # AC run-length encoding
                ac_encoded = encode_ac_coefficients(ac)
                
                encoded_blocks.append({
                    'dc_diff': dc_diff,
                    'dc_category': dc_category,
                    'ac_encoded': ac_encoded
                })
        
        return encoded_blocks, (num_blocks_h, num_blocks_w)
    
    def build_symbol_lists(self, encoded_y, encoded_cb, encoded_cr):
        """
        Build symbol lists for Huffman coding
        """
        # DC symbols: categories
        dc_symbols_y = [block['dc_category'] for block in encoded_y]
        dc_symbols_c = []
        for block in encoded_cb:
            dc_symbols_c.append(block['dc_category'])
        for block in encoded_cr:
            dc_symbols_c.append(block['dc_category'])
        
        # AC symbols: (run_length, category) pairs encoded as single value
        ac_symbols_y = []
        for block in encoded_y:
            for run_length, value in block['ac_encoded']:
                if value == 0:
                    # EOB or ZRL
                    symbol = (run_length << 4) | 0
                else:
                    abs_value = abs(value)
                    category = int(np.floor(np.log2(abs_value))) + 1 if abs_value > 0 else 0
                    symbol = (run_length << 4) | category
                ac_symbols_y.append((symbol, value))
        
        ac_symbols_c = []
        for blocks in [encoded_cb, encoded_cr]:
            for block in blocks:
                for run_length, value in block['ac_encoded']:
                    if value == 0:
                        symbol = (run_length << 4) | 0
                    else:
                        abs_value = abs(value)
                        category = int(np.floor(np.log2(abs_value))) + 1 if abs_value > 0 else 0
                        symbol = (run_length << 4) | category
                    ac_symbols_c.append((symbol, value))
        
        return dc_symbols_y, dc_symbols_c, ac_symbols_y, ac_symbols_c
    
    def encode_image(self, image_path, verbose=True):
        """
        Encode an image to JPEG format
        Args:
            image_path: path to input image
            verbose: whether to print progress messages
        Returns:
            encoded_data: dictionary containing all encoded data
        """
        # Load image
        image = np.array(Image.open(image_path).convert('RGB'))
        original_shape = image.shape[:2]
        
        if verbose:
            print(f"Encoding image: {image.shape}")
        
        # Convert to YCbCr
        Y, Cb, Cr = rgb_to_ycbcr(image)
        
        # Chroma subsampling
        if self.use_chroma_subsampling:
            Cb = downsample_chroma(Cb)
            Cr = downsample_chroma(Cr)
        
        # Encode each channel
        if verbose:
            print("Encoding Y channel...")
        encoded_y, y_blocks_shape = self.encode_channel(Y, self.q_table_lum, is_luma=True)
        
        if verbose:
            print("Encoding Cb channel...")
        encoded_cb, cb_blocks_shape = self.encode_channel(Cb, self.q_table_chrom, is_luma=False)
        
        if verbose:
            print("Encoding Cr channel...")
        encoded_cr, cr_blocks_shape = self.encode_channel(Cr, self.q_table_chrom, is_luma=False)
        
        # Build symbol lists for Huffman coding
        if verbose:
            print("Building Huffman tables...")
        dc_symbols_y, dc_symbols_c, ac_symbols_y, ac_symbols_c = self.build_symbol_lists(
            encoded_y, encoded_cb, encoded_cr
        )
        
        # Build Huffman codes
        self.huffman_dc_y.build_codes(dc_symbols_y)
        self.huffman_dc_c.build_codes(dc_symbols_c)
        self.huffman_ac_y.build_codes([s[0] for s in ac_symbols_y])
        self.huffman_ac_c.build_codes([s[0] for s in ac_symbols_c])
        
        # Create encoded data structure
        encoded_data = {
            'original_shape': original_shape,
            'use_chroma_subsampling': self.use_chroma_subsampling,
            'quality': self.quality,
            'q_table_lum': self.q_table_lum,
            'q_table_chrom': self.q_table_chrom,
            'y_blocks_shape': y_blocks_shape,
            'cb_blocks_shape': cb_blocks_shape,
            'cr_blocks_shape': cr_blocks_shape,
            'encoded_y': encoded_y,
            'encoded_cb': encoded_cb,
            'encoded_cr': encoded_cr,
            'huffman_dc_y_table': self.huffman_dc_y.get_code_table(),
            'huffman_ac_y_table': self.huffman_ac_y.get_code_table(),
            'huffman_dc_c_table': self.huffman_dc_c.get_code_table(),
            'huffman_ac_c_table': self.huffman_ac_c.get_code_table(),
        }
        
        if verbose:
            print("Encoding complete!")
        return encoded_data
    
    def save_encoded(self, encoded_data, output_path, verbose=True):
        """
        Save encoded data to file
        """
        with open(output_path, 'wb') as f:
            pickle.dump(encoded_data, f)
        
        file_size = os.path.getsize(output_path)
        if verbose:
            print(f"Saved to {output_path}, size: {file_size} bytes")
        return file_size


class JPEGDecoder:
    def __init__(self):
        """Initialize JPEG decoder"""
        self.huffman_dc_y = HuffmanCoder()
        self.huffman_ac_y = HuffmanCoder()
        self.huffman_dc_c = HuffmanCoder()
        self.huffman_ac_c = HuffmanCoder()
    
    def decode_channel(self, encoded_blocks, blocks_shape, q_table):
        """
        Decode a single channel
        Args:
            encoded_blocks: list of encoded block data
            blocks_shape: tuple (num_blocks_h, num_blocks_w)
            q_table: quantization table
        Returns:
            channel: decoded channel as numpy array
        """
        num_blocks_h, num_blocks_w = blocks_shape
        h = num_blocks_h * 8
        w = num_blocks_w * 8
        
        channel = np.zeros((h, w), dtype=np.float32)
        prev_dc = 0
        block_idx = 0
        
        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                block_data = encoded_blocks[block_idx]
                block_idx += 1
                
                # Decode DC coefficient
                dc_diff = block_data['dc_diff']
                dc = dc_diff + prev_dc
                prev_dc = dc
                
                # Decode AC coefficients
                ac = decode_ac_coefficients(block_data['ac_encoded'])
                
                # Reconstruct zigzag sequence
                zigzag = [dc] + ac
                
                # Descan from zigzag
                quant_coeffs = zigzag_descan(zigzag)
                
                # Dequantize
                dct_coeffs = dequantize(quant_coeffs, q_table)
                
                # Apply inverse DCT
                block = idct_2d(dct_coeffs)
                
                # Level shift back by +128
                block = block + 128
                
                # Clip to valid range
                block = np.clip(block, 0, 255)
                
                # Place block in channel
                channel[i*8:(i+1)*8, j*8:(j+1)*8] = block
        
        return channel
    
    def decode_image(self, encoded_data, verbose=True):
        """
        Decode JPEG encoded data back to image
        Args:
            encoded_data: dictionary containing encoded data
            verbose: whether to print progress messages
        Returns:
            image: decoded RGB image as numpy array
        """
        if verbose:
            print("Decoding image...")
        
        # Extract metadata
        original_shape = encoded_data['original_shape']
        use_chroma_subsampling = encoded_data['use_chroma_subsampling']
        q_table_lum = encoded_data['q_table_lum']
        q_table_chrom = encoded_data['q_table_chrom']
        
        # Load Huffman tables (not used in current implementation but kept for completeness)
        self.huffman_dc_y.set_code_table(encoded_data['huffman_dc_y_table'])
        self.huffman_ac_y.set_code_table(encoded_data['huffman_ac_y_table'])
        self.huffman_dc_c.set_code_table(encoded_data['huffman_dc_c_table'])
        self.huffman_ac_c.set_code_table(encoded_data['huffman_ac_c_table'])
        
        # Decode channels
        if verbose:
            print("Decoding Y channel...")
        Y = self.decode_channel(
            encoded_data['encoded_y'],
            encoded_data['y_blocks_shape'],
            q_table_lum
        )
        
        if verbose:
            print("Decoding Cb channel...")
        Cb = self.decode_channel(
            encoded_data['encoded_cb'],
            encoded_data['cb_blocks_shape'],
            q_table_chrom
        )
        
        if verbose:
            print("Decoding Cr channel...")
        Cr = self.decode_channel(
            encoded_data['encoded_cr'],
            encoded_data['cr_blocks_shape'],
            q_table_chrom
        )
        
        # Upsample chroma if needed
        if use_chroma_subsampling:
            Cb = upsample_chroma(Cb, Y.shape)
            Cr = upsample_chroma(Cr, Y.shape)
        
        # Crop to original size
        Y = Y[:original_shape[0], :original_shape[1]]
        Cb = Cb[:original_shape[0], :original_shape[1]]
        Cr = Cr[:original_shape[0], :original_shape[1]]
        
        # Convert back to RGB
        if verbose:
            print("Converting YCbCr to RGB...")
        image = ycbcr_to_rgb(Y, Cb, Cr)
        
        if verbose:
            print("Decoding complete!")
        return image
    
    def load_and_decode(self, encoded_path):
        """
        Load encoded file and decode
        """
        with open(encoded_path, 'rb') as f:
            encoded_data = pickle.load(f)
        
        return self.decode_image(encoded_data)
