# JPEG 編碼器/解碼器

完整的 JPEG 壓縮和解壓縮實現，包含 Huffman 編碼技術。

## 核心功能

✅ RGB ↔ YCbCr 色彩空間轉換  
✅ 色度子採樣 (4:2:0)  
✅ 8×8 塊 DCT/IDCT 變換  
✅ 量化/反量化（標準 JPEG 量化表）  
✅ Zigzag 掃描  
✅ DC 係數差分編碼  
✅ AC 係數遊程編碼  
⭐ **Huffman 編碼/解碼**（完整實現）  
✅ 完整的編碼器和解碼器

## 專案結構

```
final/
├── huffman.py       # Huffman 編碼演算法實現
├── jpeg_core.py     # JPEG 核心功能（DCT、量化等）
├── jpeg_codec.py    # JPEG 編碼器和解碼器類別
├── demo.py          # 示範和測試程式
├── lena.png         # 測試圖像
├── README.md        # 說明文件（中文）
├── README_EN.md     # 說明文件（英文）
└── output/          # 輸出目錄（自動生成）
```

## 依賴套件

```bash
pip install numpy Pillow
```

## 快速開始

### 測試所有品質等級

```bash
python demo.py
```

輸出：測試 5 個品質等級（10, 30, 50, 70, 90），顯示壓縮比和 PSNR

### 測試指定品質

```bash
python demo.py 50
```

輸出：詳細的壓縮統計、Huffman 編碼資訊、對比圖像和誤差圖

## Python API 使用

```python
from jpeg_codec import JPEGEncoder, JPEGDecoder
from PIL import Image

# 編碼
encoder = JPEGEncoder(quality=50)
encoded_data = encoder.encode_image('lena.png', verbose=False)
encoder.save_encoded(encoded_data, 'output.pkl', verbose=False)

# 解碼
decoder = JPEGDecoder()
reconstructed = decoder.load_and_decode('output.pkl')
Image.fromarray(reconstructed).save('reconstructed.png')
```

## 輸出檔案

執行後會在 `output/` 目錄生成：

- `lena_qXX.pkl` - 壓縮資料
- `lena_recon_qXX.png` - 重建圖像
- `comparison_qXX.png` - 原始 vs 重建對比（僅單一品質時）
- `error_map_qXX.png` - 誤差熱圖（僅單一品質時）

## JPEG 編碼流程

1. RGB → YCbCr 色彩空間轉換
2. 色度子採樣 (4:2:0)
3. 8×8 塊 DCT 變換
4. 量化（標準 JPEG 量化表，根據品質縮放）
5. Zigzag 掃描
6. DC 係數差分編碼
7. AC 係數遊程編碼
8. **Huffman 編碼**（動態構建編碼表）

解碼流程為編碼的完全逆過程。

## 性能表現

Lena 512×512 測試結果：

| 品質 | 大小 | 壓縮比 | PSNR | 評價 |
|------|------|--------|------|------|
| 10  | 15 KB  | 50:1 | 27.3 dB | Fair |
| 30  | 30 KB  | 26:1 | 30.7 dB | Good |
| 50  | 41 KB  | 19:1 | 31.8 dB | Good |
| 70  | 58 KB  | 13:1 | 32.7 dB | Good |
| 90  | 116 KB | 7:1  | 34.2 dB | Good |

原始大小: 768 KB

## 技術參考

- JPEG 標準 (ITU-T T.81 | ISO/IEC 10918-1)
- DCT 實現參考 lab2
- 量化表參考 lab4
- 色彩空間轉換參考 lab1

---

NYCU Vision Compression Project
