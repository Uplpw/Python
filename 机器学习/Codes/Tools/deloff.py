import os

path="F:\\lpw\\ModelNet_Blender_OFF2Multiview-master\\10test-1"
list=os.listdir(path)

for i in list:
    fpath=path+"\\"+i
    dir=os.listdir(fpath)
    for j in dir:
        print(fpath+"\\"+j)
        os.remove(fpath+"\\"+j)