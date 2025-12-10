"""
Huffman Encoding/Decoding Module for JPEG Compression
"""
import heapq
from collections import defaultdict
import numpy as np


class HuffmanNode:
    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right
    
    def __lt__(self, other):
        return self.freq < other.freq


class HuffmanCoder:
    def __init__(self):
        self.codes = {}
        self.reverse_codes = {}
    
    def build_frequency_table(self, symbols):
        """Build frequency table from symbols"""
        freq_table = defaultdict(int)
        for symbol in symbols:
            freq_table[symbol] += 1
        return freq_table
    
    def build_huffman_tree(self, freq_table):
        """Build Huffman tree from frequency table"""
        if not freq_table:
            return None
        
        # Create a min heap of nodes
        heap = [HuffmanNode(symbol=sym, freq=freq) for sym, freq in freq_table.items()]
        heapq.heapify(heap)
        
        # Special case: only one symbol
        if len(heap) == 1:
            node = heapq.heappop(heap)
            root = HuffmanNode(freq=node.freq, left=node)
            return root
        
        # Build tree
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            parent = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
            heapq.heappush(heap, parent)
        
        return heap[0]
    
    def generate_codes(self, root, current_code=""):
        """Generate Huffman codes from tree"""
        if root is None:
            return
        
        # Leaf node
        if root.symbol is not None:
            # Handle single symbol case
            self.codes[root.symbol] = current_code if current_code else "0"
            self.reverse_codes[current_code if current_code else "0"] = root.symbol
            return
        
        # Traverse left and right
        if root.left:
            self.generate_codes(root.left, current_code + "0")
        if root.right:
            self.generate_codes(root.right, current_code + "1")
    
    def build_codes(self, symbols):
        """Build Huffman codes from symbol list"""
        freq_table = self.build_frequency_table(symbols)
        tree = self.build_huffman_tree(freq_table)
        self.codes = {}
        self.reverse_codes = {}
        self.generate_codes(tree)
        return self.codes
    
    def encode(self, symbols):
        """Encode symbols to bitstring"""
        if not self.codes:
            raise ValueError("Huffman codes not built. Call build_codes first.")
        
        bitstring = ""
        for symbol in symbols:
            if symbol in self.codes:
                bitstring += self.codes[symbol]
            else:
                raise ValueError(f"Symbol {symbol} not in Huffman codes")
        return bitstring
    
    def decode(self, bitstring):
        """Decode bitstring to symbols"""
        if not self.reverse_codes:
            raise ValueError("Huffman codes not built. Call build_codes first.")
        
        symbols = []
        current_code = ""
        
        for bit in bitstring:
            current_code += bit
            if current_code in self.reverse_codes:
                symbols.append(self.reverse_codes[current_code])
                current_code = ""
        
        if current_code:
            # Incomplete code at the end (padding bits)
            pass
        
        return symbols
    
    def get_code_table(self):
        """Return the code table for serialization"""
        return self.codes.copy()
    
    def set_code_table(self, codes):
        """Set code table from deserialization"""
        self.codes = codes.copy()
        self.reverse_codes = {code: symbol for symbol, code in codes.items()}
