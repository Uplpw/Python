import os
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def sort(list):
    list1 = []  # length==5 (1.obj)
    list2 = []  # length==6 (10.obj)
    list3 = []  # length==7 (100.obj)
    list4 = []  # length==8 (1000.obj)
    for i in list:
        if (len(i) == 5):
            list1.append(i)
        elif (len(i) == 6):
            list2.append(i)
        elif (len(i) == 7):
            list3.append(i)
        elif (len(i) == 8):
            list4.append(i)
    sorted(list1)
    sorted(list2)
    sorted(list3)
    sorted(list4)

    return list1 + list2 + list3 + list4


def trim(list):
    trimList = []
    for i in list:
        length = len(i)
        str = i[:length - 5] + i[length - 4:]
        trimList.append(str)
    return trimList


def filerename(path):
    list = os.listdir(path)
    for i in list:
        length = len(i)
        oldname = path + "\\" + i
        newname = path + "\\" + i[:length - 5] + i[length - 4:]
        os.rename(oldname, newname)


def readobjfile(path):
    listX = []
    listY = []
    listZ = []
    f = open(path)
    str = f.readline()
    while (str != ""):
        if (str[0] == "#" or str[0] == "f"):
            pass
        else:
            str = str.strip("\n")
            list = str.split(" ")
            listX.append(float(list[1]))
            listY.append(float(list[2]))
            listZ.append(float(list[3]))
        str = f.readline()
    return listX, listY, listZ


def plot(X, Y, Z):
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(X, Y, Z, c='r')
    # surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_zlim(-1.01, 1.01)

    ax.zaxis.set_major_locator(LinearLocator(10))

    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.

    # fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


path = "F:\\lpw\\Tranfer\\fat2fit\\test\\101\\BtoA\\B"
list = os.listdir(path)

if (list[1].find("_") >= 0):
    print("file rename")
    filerename(path)

list = os.listdir(path)
list = sort(list)
for i in list:
    objpath = path + "\\" + i
    if (i == "1.obj"):
        X, Y, Z = readobjfile(objpath)
        plot(X, Y, Z)
