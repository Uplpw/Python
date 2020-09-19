import os


def off(path):
    f = open(path, 'r')
    str = f.readline()
    if (len(str) == 4):
        pass
        f.close()
        return 0
    else:
        print(path)
        f.close()
        return 1


def modify(path, topath, flag):
    if flag == 1:
        f = open(path, 'r')
        ff = open(topath, 'w')
        str = f.readline()
        ff.write(str[:3] + "\n")
        ff.write(str[3:])
        while (str != ""):
            str = f.readline()
            if (str != "\n"):
                ff.write(str)
        f.close()
        ff.close()
    else:
        f = open(path, 'r')
        ff = open(topath, 'w')
        str = f.readline()
        print(str)
        while (str != ""):
            ff.write(str)
            str = f.readline()
        f.close()
        ff.close()


path = "F:\\lpw\\ModelNet_Blender_OFF2Multiview-master\\40train-111"
topath = "F:\\lpw\\ModelNet_Blender_OFF2Multiview-master\\40ain"
list = os.listdir(path)

for i in list:
    fpath = path + "\\" + i
    dirlist = os.listdir(fpath)

    for j in dirlist:
        if off(fpath + "\\" + j) == 1:
            modify(fpath + "\\" + j, topath + "\\" + i + "\\" + j, 1)
        else:
            modify(fpath + "\\" + j, topath + "\\" + i + "\\" + j, 0)
