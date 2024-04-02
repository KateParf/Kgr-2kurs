import numpy as np
from PIL import Image

# 1-------
#H, W = map(int, input().split())
H, W = 300 , 300
#инициализация полутонового изображения размера HxW чёрными пикселями
img1 = np.zeros((H, W), dtype = np.uint8)

img_img = Image.fromarray(img1)
img_img.show()

image_filename = "demo1.jpeg"
img_img.save(image_filename)

#2-----
#инициализация полутонового изображения размера HxW белыми пикселями
img2 = np.full((H, W), 255, dtype = np.uint8)
img_img = Image.fromarray(img2)
img_img.show()

image_filename = "demo2.jpeg"
img_img.save(image_filename)

#3-----
#инициализация полутонового изображения размера HxWx3
img3 = np.full((H, W, 3), (255, 0, 0), dtype = np.uint8)
img_img = Image.fromarray(img3, mode='RGB')
img_img.show()

image_filename = "demo3.jpeg"
img_img.save(image_filename)

#4-----
#Создать матрицу размера H*W*3, заполнить её элементы произвольными значениями
img4 = np.full((H, W, 3), 0, dtype = np.uint8)
for h in range(H):
    for w in range(W):
        for i in range(3):
            img4[h, w, i] = (h+w+i) % 256

img_img = Image.fromarray(img4, mode='RGB')
img_img.show()

image_filename = "demo4.jpeg"
img_img.save(image_filename)