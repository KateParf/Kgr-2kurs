from math import sqrt
import math
import random
import numpy as np
from PIL import Image


# функция нахождения нормали и скаляра
def findScalar(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    # ищем нормаль с помощью векторного произведения
    normal = np.cross([x1-x2, y1-y2, z1-z2], [x1-x0, y1-y0, z1-z0])

    # ||n||
    norma = (sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2))

    l = [0.0, 0.0, 1.0]
    if norma != 0:
        return np.matmul(normal, l) / norma
    return 0

# функция вычисления барицентрических координат
def barycentric(x, y, x0, y0, x1, y1, x2, y2):
    if ((x1 - x2)*(y0 - y2) - (y1 - y2)*(x0 - x2) != 0): # чтоб не было ошибки с делением на 0
        lambda0 = ((x1 - x2)*(y - y2) - (y1 - y2)*(x - x2)) / ((x1 -
        x2)*(y0 - y2) - (y1 - y2)*(x0 - x2))
    else: lambda0 = 0

    if ((x2 - x0)*(y1 - y0) - (y2 - y0)*(x1 - x0) != 0): # чтоб не было ошибки с делением на 0
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
    
    # косинус угла падения направленного света для базового освещения
    cosNL = findScalar(poRot1[0], poRot1[1], poRot1[2], poRot2[0], poRot2[1], poRot2[2], poRot3[0], poRot3[1], poRot3[2])
    color = 255*cosNL

    for x in range(xmin, xmax):
        for y in range(ymin, ymax):
                lambds = barycentric(x, y, point1[0], point1[1], point2[0], point2[1], point3[0], point3[1])
                if (lambds[0] >= 0 and lambds[1] >= 0 and lambds[2] >= 0):
                    # вычисляем z-координату исходного полигона (с учетом поворота)
                    z = lambds[0]*poRot1[2] + lambds[1]*poRot2[2] + lambds[2]*poRot3[2]
                    if z <= zbuf[y][x]:
                        image[y, x] = (color, 0, 0)
                        zbuf[y][x] = z                
            

# функция для поворота точки вокруг заданных осей
def rotate(point, a, b, g, tx, ty):
    # матрицы поворота вокруг осей x, y, z
    R1 = [[1, 0, 0], [0, math.cos(a), math.sin(a)], [0, -math.sin(a), math.cos(a)]]

    R2 = [[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]]

    R3 = [[math.cos(g), math.sin(g), 0], [-math.sin(g), math.cos(g), 0], [0, 0, 1]]

    # умножаем матрицы поворота для получения итоговой матрицы поворота
    R =  np.matmul(R1, R2)
    Rfin = np.matmul(R, R3)

    # умножаем итоговую матрицу поворота на точку для получения повернутой точки
    res = np.matmul(Rfin, point)

    # добавляем сдвиг к координатам точки
    res[0] += tx
    res[1] += ty

    return res

# функция для проективного преобразования точки
def projectiveTransformation(ax, ay, x, y, u0, v0): 
    # матрица проективного преобразования
    matrix = [[ax, 0, u0],
              [0, ay, v0],
              [0, 0, 1]]
    coord = [x, y, 1]
    # получаем новую точку
    res = np.matmul(matrix, coord)
    return res

def main():

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

    # создаем пустое изображение и z-буфер
    img = np.full((1000, 1000, 3), 255, dtype=np.uint8)
    zbuffer = [[1000.0 for j in range(1000)] for i in range(1000)]

    for fa in f:
        v1, vt1, vn1 = map(int, fa[0].split("/"))
        v2, vt2, vn2 = map(int, fa[1].split("/"))
        v3, vt3, vn3 = map(int, fa[2].split("/"))

        # сдвиг 
        tx, ty = 0, -200
        # масштабирование
        s = 5000
        ax, ay = 2, -1
        # центр изображения; [0] — количество строк, а [1] — количество столбцов в массиве
        u0, v0 = img.shape[0]/2, img.shape[1]/2

        po1 = [s*v[v1-1][0], s*v[v1-1][1], s*v[v1-1][2]]
        poRot1 = rotate(po1, 0, 90, 0, tx, ty)
        point1 = projectiveTransformation(ax, ay, poRot1[0], poRot1[1], u0, v0)

        po2 = [s*v[v2-1][0], s*v[v2-1][1], s*v[v2-1][2]]
        poRot2 = rotate(po2, 0, 90, 0, tx, ty)
        point2 = projectiveTransformation(ax, ay, poRot2[0], poRot2[1],u0, v0)

        po3 = [s*v[v3-1][0], s*v[v3-1][1], s*v[v3-1][2]]
        poRot3 = rotate(po3, 0, 90, 0, tx, ty)
        point3 = projectiveTransformation(ax, ay, poRot3[0], poRot3[1],u0, v0)

        # рисуем треугольники
        drawTr(img, zbuffer, point1, point2, point3, poRot1, poRot2, poRot3)
    
    
    img_img = Image.fromarray(img)
    img_img.show()
    image_filename = "model.jpeg"
    img_img.save(image_filename)

# вызов основной программы
main()