import numpy as np
import math
from PIL import Image

#1----
# рисовать пиксели с заданным шагом, интерполируя x и y между начальным и конечным значениями
def dotted_line1(image, x0, y0, x1, y1, count, color):
    step = 1.0/count
    for t in np.arange (0, 1, step):
        x = round ((1.0 - t)*x0 + t*x1)
        y = round ((1.0 - t)*y0 + t*y1)
        image[y, x] = color

#2----
# шаг на основе расстояния между первой и последней точкой
def dotted_line2(image, x0, y0, x1, y1, color):
    count = math.sqrt((x0 - x1)**2 + (y0 - y1)**2)
    step = 1.0/count
    for t in np.arange (0, 1, step):
        x = round ((1.0 - t)*x0 + t*x1)
        y = round ((1.0 - t)*y0 + t*y1)
        image[y, x] = color

#3----
def x_loop_line(image, x0, y0, x1, y1, color):
   
    for x in range (x0, x1):
        t = (x-x0)/(x1 - x0)
        y = round ((1.0 - t)*y0 + t*y1)
        image[y, x] = color 

def x_loop_line_hotfix_1(image, x0, y0, x1, y1, color):

    # Если начальная точка правее конечной, поменяем их местами
    if x0 > x1:
        x0, x1, = x1, x0
        y0, y1 = y1, y0

    for x in range(x0, x1):
        t = (x - x0) / (x1 - x0)
        y = int(y1 * t + y0 * (1.0 - t))
        image[y, x] = color 

def x_loop_line_hotfix_2(image, x0, y0, x1, y1, color):
   
    xchange = False
     # Если изменение по x больше, чем изменение по y, поменяем местами x и y.
    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    for x in range(x0, x1):
        t = (x - x0) / (x1 - x0)
        y = int(y1 * t + y0 * (1.0 - t))
        if xchange:
            image[x, y] = color
        else:
            image[y, x] = color 

def x_loop_line_v2(image, x0, y0, x1, y1, color):
   
    xchange = False
     # Если изменение по x больше, чем изменение по y, поменяем местами x и y.
    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    # Если начальная точка правее конечной, поменяем их местами
    if x0 > x1:
        x0, x1, = x1, x0
        y0, y1 = y1, y0

    for x in range(x0, x1):
        t = (x - x0) / (x1 - x0)
        y = int(y1 * t + y0 * (1.0 - t))
        if xchange:
            image[x, y] = color
        else:
            image[y, x] = color    

#4----
def x_loop_line_v2_no_y_calc(image, x0, y0, x1, y1, color):
   
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    y = y0
    dy = abs((y1 - y0) / (x1 - x0))
    derror = 0.0 #погрешность
    y_update = 1 if y1 > y0 else -1

    for x in range(x0, x1):
        
        if xchange:
            image[x, y] = color
        else:
            image[y, x] = color

        derror += dy
        if derror > 0.5:
            derror -= 1.0
            y += y_update

def x_loop_line_v2_no_y_calc_with_some_unc_reasons(image, x0, y0, x1, y1, color):
   
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    y = y0
    dy = 2.0*(x1 - x0)*abs(y1 - y0)/(x1 - x0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1

    for x in range(x0, x1):
        
        if xchange:
            image[x, y] = color
        else:
            image[y, x] = color

        derror += dy
        if (derror > 2.0*(x1 - x0)*0.5):
            derror -= 2.0*(x1 - x0)*1.0
            y += y_update

#5----
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


def drav_and_save():
    img1 = np.zeros((200, 200, 3), dtype = np.uint8)
    img2 = np.zeros((200, 200, 3), dtype = np.uint8)
    img3 = np.zeros((200, 200, 3), dtype = np.uint8)
    img4 = np.zeros((200, 200, 3), dtype = np.uint8)
    img5 = np.zeros((200, 200, 3), dtype = np.uint8)
    img6 = np.zeros((200, 200, 3), dtype = np.uint8)
    img7 = np.zeros((200, 200, 3), dtype = np.uint8)
    img8 = np.zeros((200, 200, 3), dtype = np.uint8)
    img9 = np.zeros((200, 200, 3), dtype = np.uint8)
    for i in range(0, 12):
        x0, y0 = 100, 100
        al = (2*i*math.pi)/13
        x1 = round(100 + 95*math.cos(al))
        y1 = round(100 + 95*math.sin(al))
        dotted_line1(img1, x0, y0, x1, y1, 100, 255)
        dotted_line2(img2, x0, y0, x1, y1, 255)
        x_loop_line(img3, x0, y0, x1, y1, 255)
        x_loop_line_hotfix_1(img4, x0, y0, x1, y1, 255)
        x_loop_line_hotfix_2(img5, x0, y0, x1, y1, 255)
        x_loop_line_v2(img6, x0, y0, x1, y1, 255)
        x_loop_line_v2_no_y_calc(img7, x0, y0, x1, y1, 255)
        x_loop_line_v2_no_y_calc_with_some_unc_reasons(img8, x0, y0, x1, y1, 255)
        bez_loop_line(img9, x0, y0, x1, y1, 255)

    arr_imgs = [img1, img2, img3, img4, img5, img6, img7, img8, img9]
    for i in range(1, len(arr_imgs)+1):
        img_img = Image.fromarray(arr_imgs[i-1])
        img_img.show()
        image_filename = "demo" + str(i) + ".jpeg"
        img_img.save(image_filename)


drav_and_save()