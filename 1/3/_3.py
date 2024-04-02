import numpy as np
from PIL import Image

file = open("model_1.obj")
v = []
f = []

for s in file:
    el = list(map(str, s.split(" ")))
    key = el[0]
    if key == "v":   
        x, y, z = el[1], el[2], el[3]
        v.append([float(x), float(y), float(z)])
    if key == "f":
        f1, f2, f3 = el[1], el[2], el[3]
        f.append([f1, f2, f3])

# Нарисовать вершины модели (игнорируя координату Z) на изображении размером (1000, 1000).   
img1 = np.zeros((1000, 1000, 3), dtype = np.uint8)
for ver in v:
    img1[round(5000*ver[0] + 500), round(5000*ver[1] + 500)] = 255

img_img = Image.fromarray(img1)
img_img.show()
image_filename = "model.jpeg"
img_img.save(image_filename)

#Отрисовать все рёбра всех полигонов модели с помощью алгоритма Брезенхема
def bez_loop_line(image, x0, y0, x1, y1, color):
   
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    y = y0
    dy = 2*abs(y1 - y0)
    derror = 0.0 #погрешность
    y_update = 1 if y1 > y0 else -1

    for x in range(x0, x1):
        derror += dy

        if xchange:
            image[x, y] = color
        else:
            image[y, x] = color

        if derror > (x1 - x0) :
            derror -= 2*(x1 - x0)
            y += y_update


img2 = np.zeros((1000, 1000, 3), dtype = np.uint8)
for fa in f:
   v1, vt1, vn1 = map(int, fa[0].split("/"))
   v2, vt2, vn2 = map(int, fa[1].split("/"))
   v3, vt3, vn3 = map(int, fa[2].split("/"))

   x1, y1 = round(5000*v[v1-1][0] + 500), round(5000*v[v2-1][1] + 500) 
   x2, y2 = round(5000*v[v2-1][0] + 500), round(5000*v[v3-1][1] + 500) 
   x3, y3 = round(5000*v[v3-1][0] + 500), round(5000*v[v3-1][1] + 500) 
   bez_loop_line(img2, x1, y1, x2, y2, 255)
   bez_loop_line(img2, x1, y1, x3, y3, 255)
   bez_loop_line(img2, x2, y2, x3, y3, 255)

  
   
img_img = Image.fromarray(img2)
img_img.show()
image_filename = "model2.jpeg"
img_img.save(image_filename)