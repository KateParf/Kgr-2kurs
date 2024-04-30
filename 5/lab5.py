from math import sqrt
import math
from pyquaternion import Quaternion
import numpy as np
from PIL import Image

# функция нахождения нормали 
def findNormal(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    normal = np.cross([x1-x2, y1-y2, z1-z2], [x1-x0, y1-y0, z1-z0])
    return normal

# функция нахождения скаляра
def findScalar(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    # ищем нормаль с помощью векторного произведения
    normal = findNormal(x0, y0, z0, x1, y1, z1, x2, y2, z2)

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
def drawTr(image, gouraud, point1, point2, point3, poRot1, poRot2, poRot3, vtpo1, vtpo2, vtpo3):

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
    color = -255*cosNL

    for x in range(xmin, xmax):
        for y in range(ymin, ymax):
                lambds = barycentric(x, y, point1[0], point1[1], point2[0], point2[1], point3[0], point3[1])
                if (lambds[0] >= 0 and lambds[1] >= 0 and lambds[2] >= 0):
                    # вычисляем значение яркости (цвета) пикселя
                    I = -225*(lambds[0]*gouraud[0] + lambds[1]*gouraud[1] + lambds[2]*gouraud[2])
                    # вычисляем значение цвета пиксела в изображении текстуры
                    w = math.ceil(wt*(lambds[0]*vtpo1[0] + lambds[1]*vtpo2[0] + lambds[2]*vtpo3[0]))
                    h = math.ceil(ht*(lambds[0]*vtpo1[1] + lambds[1]*vtpo2[1] + lambds[2]*vtpo3[1]))
                    # вычисляем z-координату исходного полигона (с учетом поворота)
                    z = lambds[0]*poRot1[2] + lambds[1]*poRot2[2] + lambds[2]*poRot3[2]
                    if z <= zbuffer[y][x]:
                        #image[y, x] = (I, 0, 0)
                        image[y, x] = imgTextures.getpixel((w, h))
                        zbuffer[y][x] = z                
            
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

# поворот с помощью кватерниона 
def quaternionsConversion(ux, uy, uz, t):
    """q = Quaternion(np.array([math.cos(t/2), ux*math.sin(t/2), uy*math.sin(t/2), uz*math.sin(t/2)]))
    qq = Quaternion(np.array([math.cos(t/2), -ux*math.sin(t/2), -uy*math.sin(t/2), -uz*math.sin(t/2)]))
    p = Quaternion(np.array([0, x, y, z]))
    pp = q*p*qq"""

    a = math.cos(t/2)
    b = ux*math.sin(t/2)
    c = uy*math.sin(t/2)
    d = uz*math.sin(t/2)
    
    # матрица поворота
    R = [[0, 0, 0] for i in range(3)]

    R[0][0] = a**2 + b**2 - c**2 - d**2
    R[0][1] = 2*b*c - 2*a*d
    R[0][2] = 2*b*d + 2*a*c

    R[1][0] = 2*b*c + 2*a*d
    R[1][1] = a**2 - b**2 + c**2 - d**2
    R[1][2] = 2*c*d - 2*a*b

    R[2][0] = 2*b*d - 2*a*c
    R[2][1] = 2*c*d - 2*a*b
    R[2][2] = a**2 - b**2 - c**2 + d**2

    return R
    
# функция для поворота точки вокруг заданных осей
def rotateQuaternions(point, u, t, tx, ty):

    # получаем матрицу поворота с помощью кватерниона
    Rfin = quaternionsConversion(u[0], u[1], u[2], t)

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

# функция затенения Гуро
def darkeningGuro(normals):
    l = [0.0, 0.0, 1.0]

    normal1 = normals[0]
    normal2 = normals[1]
    normal3 = normals[2]
    
    I1 = np.dot(normal1, l) / (np.linalg.norm(normal1))
    I2 = np.dot(normal2, l) / (np.linalg.norm(normal2))
    I3 = np.dot(normal3, l) / (np.linalg.norm(normal3))
    return [I1, I2, I3]

# подсчет нормалей для каждого полигона перед отрисовкой
def allNormals(v, f):
    normals = [[0, 0, 0]]*len(v)
    for i in f:
        n = findNormal(v[i[0] - 1][0], v[i[0] - 1][1], v[i[0] - 1][2], 
                       v[i[1] - 1][0], v[i[1] - 1][1], v[i[1] - 1][2], 
                       v[i[2] - 1][0], v[i[2] - 1][1], v[i[2] - 1][2])
        for j in range(3):
             normals[i[j] - 1] += n
    return normals

# файл модели
file = open("model_1.obj")
# текстуры
imgTextures = Image.open("bunny-atlas.jpg") 
# Flip the original image vertically
imgTextures = imgTextures.transpose(method=Image.FLIP_TOP_BOTTOM)

v = []
vt = []
f = []
ffornorms = []
# парсинг
for s in file:
    el = list(map(str, s.split(" ")))
    key = el[0]
    if key == "v":   
        x, y, z = el[1], el[2], el[3]
        v.append([float(x), float(y), float(z)])
    if key == "vt":   
        vtx, vty = el[1], el[2]
        vt.append([float(vtx), float(vty)])
    if key == "f":
        f1, f2, f3 = el[1], el[2], el[3]
        f.append([f1, f2, f3])

        f1, f2, f3 = el[1].split("/"), el[2].split("/"), el[3].split("/")
        f1[0], f1[1], f1[2] = int(f1[0]), int(f1[1]), int(f1[2])
        f2[0], f2[1], f2[2] = int(f2[0]), int(f2[1]), int(f2[2])
        f3[0], f3[1], f3[2] = int(f3[0]), int(f3[1]), int(f3[2])
        ffornorms.append([f1[0], f2[0], f3[0]])

# создаем пустое изображение и z-буфер
img = np.full((1000, 1000, 3), 255, dtype=np.uint8)
zbuffer = [[1500.0 for j in range(1000)] for i in range(1000)]
normalsForPoints = allNormals(v, ffornorms)
# сдвиг 
tx, ty = 0, -200
# масштабирование
s = 5000
ax, ay = 1, -1
# центр изображения; [0] — количество строк, а [1] — количество столбцов в массиве
u0, v0 = img.shape[0]/2, img.shape[1]/2
# высота и ширина рисунка с текстурами
wt = imgTextures.width
ht = imgTextures.height

#другой поворот
for fa in f:
    v1, vt1, vn1 = map(int, fa[0].split("/"))
    v2, vt2, vn2 = map(int, fa[1].split("/"))
    v3, vt3, vn3 = map(int, fa[2].split("/"))

    # нормали для текущих точек
    normals = [normalsForPoints[v1 - 1], normalsForPoints[v2 - 1], normalsForPoints[v3 - 1]]
    u = [-0.5, 1, 0.5]
    t = 78
    po1 = [s*v[v1-1][0], s*v[v1-1][1], s*v[v1-1][2]]
    vtpo1 = [vt[vt1-1][0], vt[vt1-1][1]]
    poRot1 = rotateQuaternions(po1, u, t, tx, ty)
    point1 = projectiveTransformation(ax, ay, poRot1[0], poRot1[1], u0, v0)

    po2 = [s*v[v2-1][0], s*v[v2-1][1], s*v[v2-1][2]]
    vtpo2 = [vt[vt2-1][0], vt[vt2-1][1]]
    poRot2 = rotateQuaternions(po2, u, t, tx, ty)
    point2 = projectiveTransformation(ax, ay, poRot2[0], poRot2[1], u0, v0)

    po3 = [s*v[v3-1][0], s*v[v3-1][1], s*v[v3-1][2]]
    vtpo3 = [vt[vt3-1][0], vt[vt3-1][1]]
    poRot3 = rotateQuaternions(po3, u, t, tx, ty)
    point3 = projectiveTransformation(ax, ay, poRot3[0], poRot3[1], u0, v0)       

    gouraud = darkeningGuro(normals)
    # рисуем треугольники
    drawTr(img, gouraud, point1, point2, point3, poRot1, poRot2, poRot3, vtpo1, vtpo2, vtpo3)

img_img = Image.fromarray(img)
img_img.show()
image_filename = "model.jpeg"
img_img.save(image_filename)
