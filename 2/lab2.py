import random
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



# функция вычисления барицентрических координат
def barycentric(x, y, x0, y0, x1, y1, x2, y2):
    lambda0 = ((x1 - x2)*(y - y2) - (y1 - y2)*(x - x2)) / ((x1 -
    x2)*(y0 - y2) - (y1 - y2)*(x0 - x2))
    lambda1 = ((x2 - x0)*(y - y0) - (y2 - y0)*(x - x0)) / ((x2 -
    x0)*(y1 - y0) - (y2 - y0)*(x1 - x0))
    lambda2 = 1.0 - lambda0 - lambda1
    return [lambda0, lambda1, lambda2]

# функцию отрисовки треугольника с вершинами (x0, y0), (x1, y1) и (x2, y2)
def drawTr(image, x0, y0, x1, y1, x2, y2):
    xmin = round(min(x0, x1, x2))
    if (xmin < 0): xmin = 0
    xmax = round(max(x0, x1, x2))

    ymin = round(min(y0, y1, y2))
    if (ymin < 0): ymin = 0
    ymax = round(max(y0, y1, y2))

    # рандомный цвет
    rnd1 = random.randint(0, 255)
    rnd2 = random.randint(0, 255)
    rnd3 = random.randint(0, 255)
    color = (rnd1, rnd2, rnd3)

    for x in range(xmin, xmax):
        for y in range(ymin, ymax):
            if ( ((x1 -x2)*(y0 - y2) - (y1 - y2)*(x0 - x2)) != 0 and  ((x2 - x0)*(y1 - y0) - (y2 - y0)*(x1 - x0)) != 0 ):

                    lambds = barycentric(x, y, x0, y0, x1, y1, x2, y2)
                    if (lambds[0] < 0 and lambds[1] < 0 and lambds[2] < 0):
                        continue         

                    image[y, x] = color 
  

img = np.zeros((1000, 1000, 3), dtype = np.uint8)
for fa in f:
   v1, vt1, vn1 = map(int, fa[0].split("/"))
   v2, vt2, vn2 = map(int, fa[1].split("/"))
   v3, vt3, vn3 = map(int, fa[2].split("/"))

   x1, y1 = 5000*v[v1-1][0] + 500, 5000*v[v2-1][1] + 300
   x2, y2 = 5000*v[v2-1][0] + 500, 5000*v[v3-1][1] + 300
   x3, y3 = 5000*v[v3-1][0] + 500, 5000*v[v3-1][1] + 300
   drawTr(img, x1, y1, x2, y2, x3, y3)

  
   
img_img = Image.fromarray(img, mode = 'RGB')
img_img = img_img.rotate(180, expand=True)
img_img.show()
image_filename = "model1.jpeg"
img_img.save(image_filename)