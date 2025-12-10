from PIL import Image
import numpy as np
import os

image = np.array(Image.open('lena.png').convert('RGB'))


r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]

rgb_file = 'rgb'
os.makedirs(rgb_file, exist_ok=True)
Image.fromarray(r).save(os.path.join(rgb_file, 'r.png'))
Image.fromarray(g).save(os.path.join(rgb_file, 'g.png'))
Image.fromarray(b).save(os.path.join(rgb_file, 'b.png'))

# Y, U, V
y = 0.299*r + 0.587*g + 0.114*b
u = -0.147*r - 0.287*g + 0.436*b
v = 0.615*r - 0.515*g - 0.100*b

# show each channel image
yuv_file = 'yuv'
os.makedirs(yuv_file, exist_ok=True)
Image.fromarray(y.astype(np.uint8)).save(os.path.join(yuv_file, 'y.png'))
Image.fromarray((u + 128).astype(np.uint8)).save(os.path.join(yuv_file, 'u.png'))
Image.fromarray((v + 128).astype(np.uint8)).save(os.path.join(yuv_file, 'v.png'))


# Y, CB, CR
Y = 0.257*r + 0.504*g + 0.098*b + 16
CB = -0.148*r - 0.291*g + 0.439*b + 128
CR = 0.439*r - 0.368*g - 0.071*b + 128

ycbcr_file = 'ycbcr'
os.makedirs(ycbcr_file, exist_ok=True)
Image.fromarray(Y.astype(np.uint8)).save(os.path.join(ycbcr_file, 'y.png'))
Image.fromarray(CB.astype(np.uint8)).save(os.path.join(ycbcr_file, 'cb.png'))
Image.fromarray(CR.astype(np.uint8)).save(os.path.join(ycbcr_file, 'cr.png'))
