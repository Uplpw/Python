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

path="F:\\lpw\\ModelNet_Blender_OFF2Multiview-master\\40test"
list=os.listdir(path)

for i in list:
    fpath=path+"\\"+i
    dir=os.listdir(fpath)

    for j in dir:
        off(fpath+"\\"+j)