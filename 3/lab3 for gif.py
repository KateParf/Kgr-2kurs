from math import sqrt
import math
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


# определяем функцию для поворота точки вокруг заданных осей
def rotate(point, a, b, g, tx, ty):
    # создаем матрицы поворота вокруг осей x, y, z
    R1 = [[1, 0, 0], [0, math.cos(a), math.sin(a)], [0, -math.sin(a), math.cos(a)]]

    R2 = [[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]]

    R3 = [[math.cos(g), math.sin(g), 0], [-math.sin(g), math.cos(g), 0], [0, 0, 1]]

    # умножаем матрицы поворота для получения итоговой матрицы поворота
    R =  np.matmul(R1, R2)

    Rfin = np.matmul(R, R3)

    xyz = [point[0], point[1], point[2]]
    # умножаем итоговую матрицу поворота на точку для получения повернутой точки
    res = np.matmul(Rfin, xyz)

    # добавляем сдвиг к координатам точки
    res[0] += tx
    res[1] += ty

    return res

def projectiveTransformation(ax, ay, x, y, z, img): 
    # создаем матрицу проективного преобразования
    matrix = [[ax, 0, img.shape[1] / 2],
              [0, ay, img.shape[0] / 2],
              [0, 0, 1]]
    coord = [x, y, 1]
    # умножаем матрицу проективного преобразования на точку для получения преобразованной точки
    res = np.dot(matrix, coord)
    return res

# функция вычисления нормали
def findNormal(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    i = [1.0, 0.0, 0.0]
    j = [0.0, 1.0, 0.0]
    k = [0.0, 0.0, 1.0]

    #n = [i[0] *((y1 - y0)*(z1-z2) - (z1-z0)*(y1-y2)), 
    #     j[1]*((x1-x0)*(z1-z2) - (z1-z0)*(x1-x2)), 
    #     k[2]*((x1-x0)*(y1-y2) - (y1-y0)*(x1-x2))]
    
    n = np.cross(np.array([x1-x2, y1-y2, z1-z2]), np.array([x1-x0, y1-y0, z1-z0]))

    return n

def scalar(x0, y0, z0, x1, y1, z1, x2, y2, z2):
     # координаты нормали
    normal = findNormal(x0, y0, z0, x1, y1, z1, x2, y2, z2)
    # ||n||
    #norma = (sqrt(n[0]**2 + n[1]**2 + n[2]**2))
    norma = np.linalg.norm(normal)
    l = [0.0, 0.0, 1.0]
    if norma != 0: 
        return np.dot(normal, l) / norma



# функция вычисления барицентрических координат
def barycentric(x, y, x0, y0, x1, y1, x2, y2):
    if ((x1 - x2)*(y0 - y2) - (y1 - y2)*(x0 - x2) != 0):
        lambda0 = ((x1 - x2)*(y - y2) - (y1 - y2)*(x - x2)) / ((x1 -
        x2)*(y0 - y2) - (y1 - y2)*(x0 - x2))
    else: lambda0 = 0

    if ((x2 - x0)*(y1 - y0) - (y2 - y0)*(x1 - x0) != 0):
        lambda1 = ((x2 - x0)*(y - y0) - (y2 - y0)*(x - x0)) / ((x2 -
        x0)*(y1 - y0) - (y2 - y0)*(x1 - x0))
    else: lambda1 = 0

    lambda2 = 1.0 - lambda0 - lambda1
    return [lambda0, lambda1, lambda2]

# функцию отрисовки треугольника 
def drawTr(image, zbuf, point1, point2, point3, poRot1, poRot2, poRot3):

    xmin = math.floor(min(point1[0], point2[0], point3[0]))
    if (xmin < 0): xmin = 0
    xmax = math.ceil(max(point1[0], point2[0], point3[0]))
    if (xmax > len(image)): xmax = len(image)
    ymin = math.floor(min(point1[1], point2[1], point3[1]))
    if (ymin < 0): ymin = 0
    ymax = math.ceil(max(point1[1], point2[1], point3[1]))
    if (ymax > len(image)): ymax = len(image)
    
    cosNL = scalar(poRot1[0], poRot1[1], poRot1[2], poRot2[0], poRot2[1], poRot2[2], poRot3[0], poRot3[1], poRot3[2])
    # рандомный цвет
    rnd1 = random.randint(0, 255)
    rnd2 = random.randint(0, 200)
    rnd3 = random.randint(0, 100)
    color = (rnd1*cosNL, rnd2*cosNL, rnd3*cosNL)


    for x in range(xmin, xmax):
        for y in range(ymin, ymax):
                lambds = barycentric(x, y, point1[0], point1[1], point2[0], point2[1], point3[0], point3[1])
                if (lambds[0] >= 0 and lambds[1] >= 0 and lambds[2] >= 0):
                    z = lambds[0]*poRot1[2] + lambds[1]*poRot2[2] + lambds[2]*poRot3[2]
                    if z <= zbuf[y][x]:
                        image[y, x] = color
                        zbuf[y][x] = z                
            
  

ugli = [0, 45, 90, 135, 180]

for ug1 in ugli:
    for ug2 in ugli:
        for ug3 in ugli:
            # создаем пустое изображение и z-буфер
            img = np.full((1000, 1000, 3), 255, dtype=np.uint8)
            zbuffer = [[1500.0 for j in range(1000)] for i in range(1000)]

            for fa in f:
                v1, vt1, vn1 = map(int, fa[0].split("/"))
                v2, vt2, vn2 = map(int, fa[1].split("/"))
                v3, vt3, vn3 = map(int, fa[2].split("/"))

                #сдвиг 
                tx, ty = 0, -200

                ax, ay = 1, -1

                po1 = [5000*v[v1-1][0], 5000*v[v1-1][1], 5000*v[v1-1][2]]
                poRot1 = rotate(po1, ug1, ug2, ug3, tx, ty)
                point1 = projectiveTransformation(ax, ay, poRot1[0], poRot1[1], poRot1[2], img)

                po2 = [5000*v[v2-1][0], 5000*v[v2-1][1], 5000*v[v2-1][2]]
                poRot2 = rotate(po2, ug1, ug2, ug3, tx, ty)
                point2 = projectiveTransformation(ax, ay, poRot2[0], poRot2[1], poRot2[2], img)

                po3 = [5000*v[v3-1][0], 5000*v[v3-1][1], 5000*v[v3-1][2]]
                poRot3 = rotate(po3, ug1, ug2, ug3, tx, ty)
                point3 = projectiveTransformation(ax, ay, poRot3[0], poRot3[1], poRot3[2], img)


                drawTr(img, zbuffer, point1, point2, point3, poRot1, poRot2, poRot3)
            
            
            img_img = Image.fromarray(img)
            #img_img.show()
            image_filename = "model"+ str(ug1) + str(ug2) + str(ug3)+ "x.jpeg"
            img_img.save(image_filename)