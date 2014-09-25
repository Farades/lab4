#!/bin/python
# -*- coding:utf-8 -*-

from PIL import Image, ImageDraw
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from colorsys import *
import numpy as np

def abs_selection( sel ):
    return [ sel[0], sel[1], sel[0]+sel[2], sel[1]+sel[3] ]

def e_kernel(r):
    return 0.75 * ( 1.0 - r**2 ) if abs(r) <= 1 else 0

# вычисление расстояния
# A, B - вектора размерности N
# если p заданно, оно должно быть целым числом
# p - критерий пространства (p==2 - эвклидово)
# W - вектор весов компонентов координат (размерность N)
# если W не задан, заполняется 1
def dist(A, B, p=None, W=None):
    s = 0
    if not W:
        ln = len(zip(A,B))
        W = [ 1 for i in range(ln) ]
    if not p:
        p = 2
        
    for a, b, w in zip(A,B,W):
        s += w * abs( a - b ) ** p

    return s ** ( 1.0 / p )

# интеграл V
# pnt - оцениваемый вектор
# X - массив векторов класса
def V(pnt, X, h, kernel):
    s = 0
    for x in X:
        s += kernel( dist(pnt, x) / h )
    return s

# pnt - оцениваемый вектор
# y - класс (массив векторов) для которого считается критерий
# Y - все классы в одном массиве
def parzen(pnt, y, Y, h, kernel):
    ly = 1.0
    zn = ly * V(pnt, Y, h, kernel)
    if zn == 0: return 0.0
    return V(pnt, y, h, kernel) / zn

def open_img_draw(img_path):
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    return (img, draw)

def open_img(img_path):
    img = Image.open(img_path)
    return img

# вывод на экран выделенной области
def select_image_rect( img, sel):
    draw = ImageDraw.Draw(img)
    w = img.size[0]
    h = img.size[1]
    pix = img.load()

    select = (sel[0] if sel[0] >= 0 else 0,
              sel[2] if sel[2] <= w else w,
              sel[1] if sel[1] >= 0 else 0,
              sel[3] if sel[3] <= h else h)

    for i in range(select[0],select[1]):
        for j in range(select[2],select[3]):
            r,g,b = img.getpixel((i,j))
            g = int( (g * 3.0 + b * 2.0 + r) / 6.0 ) * 2
            draw.point((i,j), (g,g,g))

    img.show()

# pix - массив двумерный пикселей
# win - окно в абсолютных
def calc_distributons( pix, win ):
    exp = [0.0, 0.0]
    k = 0
    for i in range(win[0], win[2]):
        for j in range(win[1], win[3]):
            h,s,v = rgb_to_hsv( pix[i,j][0], pix[i,j][1], pix[i,j][2] )
            exp[0] += h
            exp[1] += v
            k += 1
    exp[0] /= k
    exp[1] /= k

    var = [0.0, 0.0]
    for i in range(win[0], win[2]):
        for j in range(win[1], win[3]):
            h,s,v = rgb_to_hsv( pix[i,j][0], pix[i,j][1], pix[i,j][2] )
            var[0] += ( exp[0] - h ) ** 2
            var[1] += ( exp[1] - v ) ** 2
    var[0] /= (k-1)
    var[1] /= (k-1)

    return ( exp[0], var[0], exp[1], var[1] )

# подсчитать значения МО и дисперсии для
# каждого фрагмента выделеного изображения
def calc_class_sample( img, sel, step=4, winsize=8 ):
    pix = img.load()

    w = img.size[0]
    h = img.size[1]

    select = (sel[0] if sel[0] >= 0 else 0,
              sel[2] if sel[2] <= w-winsize else w-winsize,
              sel[1] if sel[1] >= 0 else 0,
              sel[3] if sel[3] <= h-winsize else h-winsize)

    res = []

    for i in range(select[0],select[1],step):
        for j in range(select[2],select[3],step):
            win = abs_selection( [i,j,winsize,winsize] )
            res.append( calc_distributons( pix, win ) )

    return res

def flat_one_level(arr):
    res = []
    for k in arr:
        res += k
    return res

colors = [ (1,0,0),
           (0,1,0),
           (0,0,1),
           (1,1,0),
           (1,0,1),
           (0,1,1) ]

colors = [ tuple( int(c*255) for c in clr ) for clr in colors ]

def prepare( img, winsize ):
    pix = img.load()
    w = img.size[0]
    h = img.size[1]
    img_res = Image.new( 'RGB', (w-winsize,h-winsize), (0,0,0) )
    draw = ImageDraw.Draw(img_res)
    return (pix,w,h,img_res,draw)

def classificate( img, winsize, Y, h, kfunc=e_kernel ):

    pix, w, h, img_res, draw = prepare(img,winsize)
    step = 4

    fY = flat_one_level(Y)
    for j in range(0,h-winsize,step):
        for i in range(0,w-winsize,step):
            win = abs_selection( [i,j,winsize,winsize] )
            pnt = calc_distributons(pix,win)
            r = [ parzen(pnt, y, fY, h, kfunc ) for y in Y ]
            clr = (0,0,0)
            maxr = max(r)
            if( maxr > 0 ):
                index = r.index(maxr)
                clr = colors[index]
            draw.rectangle( [(i,j),(i+step,j+step)], clr )
    return img_res

# clss - массив из классов, где класс это массив картежей
# картеж этот состоит из имени файла и области выделения
def get_all_classes_samples( clss, step, winsize ):
    res = []
    for class_examples in clss:
        class_sample = []
        for fname, selection in class_examples:
            image = open_img( fname )
            sel = abs_selection(selection)
            class_sample += calc_class_sample( image, sel, step, winsize )
            #select_image_rect( image, sel )
        res.append( class_sample )
    return res

classes = [
            #forest
            [ ("img/23.jpg", [150,70,100,30] ), 
              ("img/11.jpg", [180,60,90,40] ), 
              ("img/53.png",[0,0,360,217]),
              ("img/54.png",[0,0,150,245]),
            ],

            #road
            [ ("img/11.jpg", [220,174,29,24] ), 
              ("img/33.jpg", [85,137,70,70] ), 
              ("img/54.png",[181,0,104,246]), 
              ("img/43.jpg",[67,121,70,70]),
              ("img/43.jpg",[115,94,50,40]) 
            ]
          ]

def write_class_to_file( fname, cls ):
    with open(fname, "w") as f:
        for sample in cls:
            f.write( "%f %f %f %f\n"%sample )

step = 16
winsize = 16
Y = get_all_classes_samples( classes, step, winsize )
h = 6

write_class_to_file( "result_forest", Y[0] )
write_class_to_file( "result_road", Y[1] )

image = open_img( "img/56.png" )

image.show()
res = classificate( image, winsize, Y, h )
res.show()

print('complite')
