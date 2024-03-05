from math import sqrt
import random
import numpy as np
from PIL import Image

file = open("model_1.obj")
v = []
f = []
# парсинг
for s in file:
    el = list(map(str, s.split(" ")))
    key = el[0]
    if key == "v":   
        x, y, z = el[1], el[2], el[3]
        v.append([float(x), float(y), float(z)])
    if key == "f":
        f1, f2, f3 = el[1], el[2], el[3]
        f.append([f1, f2, f3])

zbuf = [[1500.0 for j in range(2000)] for i in range(2000)]

# функция вычисления нормали
def normal(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    i = [1.0, 0.0, 0.0]
    j = [0.0, 1.0, 0.0]
    k = [0.0, 0.0, 1.0]

    n = [i[0] *((y1 - y0)*(z1-z2) - (z1-z0)*(y1-y2)), 
         j[1]*((x1-x0)*(z1-z2) - (z1-z0)*(x1-x2)), 
         k[2]*((x1-x0)*(y1-y2) - (y1-y0)*(x1-x2))]
    return n
  
# функция вычисления барицентрических координат
def barycentric(x, y, x0, y0, x1, y1, x2, y2):
    lambda0 = ((x1 - x2)*(y - y2) - (y1 - y2)*(x - x2)) / ((x1 -
    x2)*(y0 - y2) - (y1 - y2)*(x0 - x2))
    lambda1 = ((x2 - x0)*(y - y0) - (y2 - y0)*(x - x0)) / ((x2 -
    x0)*(y1 - y0) - (y2 - y0)*(x1 - x0))
    lambda2 = 1.0 - lambda0 - lambda1
    return [lambda0, lambda1, lambda2]

# функцию отрисовки треугольника с вершинами (x0, y0, z0), (x1, y1, z1) и (x2, y2, z2)
def drawTr(image, x0, y0, z0, x1, y1, z1, x2, y2, z2):

    xmin = round(min(x0, x1, x2))
    if (xmin < 0): xmin = 0
    xmax = round(max(x0, x1, x2))
    if (xmax > 2000): xmax = 2000
    ymin = round(min(y0, y1, y2))
    if (ymin < 0): ymin = 0
    ymax = round(max(y0, y1, y2))
    if (ymax > 2000): ymax = 2000

    # рандомный цвет
    #rnd1 = random.randint(0, 255)
    #rnd2 = random.randint(0, 255)
    #rnd3 = random.randint(0, 255)
    #color = (rnd1, rnd2, rnd3)

    for x in range(xmin, xmax):
        for y in range(ymin, ymax):

            # координаты нормали
            n = normal(x0, y0, z0, x1, y1, z1, x2, y2, z2)
            # ||n||
            nn = (sqrt(n[0]**2 + n[1]**2 + n[2]**2))
            if nn == 0: continue
            l = [0.0, 0.0, 1.0]
            cosNL = -(n[0] * l[0] + n[1]*l[1] + n[2]*l[2]) / nn
            if ( cosNL < 0):     
                if ( ((x1 -x2)*(y0 - y2) - (y1 - y2)*(x0 - x2)) != 0 and  ((x2 - x0)*(y1 - y0) - (y2 - y0)*(x1 - x0)) != 0 ):
                    lambds = barycentric(x, y, x0, y0, x1, y1, x2, y2)
                    if (lambds[0] < 0 and lambds[1] < 0 and lambds[2] < 0):
                        continue         
                    z = lambds[0]*z0 + lambds[1]*z1 + lambds[2]*z2
                    if(z < zbuf[y][x]):  
                        image[y, x] = (-255*cosNL, 0, 0) 
                        zbuf[y][x] = z                
            
  

img = np.zeros((2000, 2000, 3), dtype = np.uint8)
for fa in f:
   v1, vt1, vn1 = map(int, fa[0].split("/"))
   v2, vt2, vn2 = map(int, fa[1].split("/"))
   v3, vt3, vn3 = map(int, fa[2].split("/"))

   x1, y1, z1 = 10000*v[v1-1][0] + 500, 10000*v[v1-1][1] + 300, 10000*v[v1-1][2] + 300
   x2, y2, z2 = 10000*v[v2-1][0] + 500, 10000*v[v2-1][1] + 300, 10000*v[v2-1][2] + 300
   x3, y3, z3 = 10000*v[v3-1][0] + 500, 10000*v[v3-1][1] + 300, 10000*v[v3-1][2] + 300
   drawTr(img, x1, y1, z1, x2, y2, z2, x3, y3, z3)
  
   
img_img = Image.fromarray(img, mode = 'RGB')
img_img = img_img.rotate(180, expand=True)
img_img.show()
image_filename = "model3.jpeg"
img_img.save(image_filename)