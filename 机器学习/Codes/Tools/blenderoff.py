import os
import shutil
path="F:\\lpw\\ModelNet_Blender_OFF2Multiview-master\\ModelNet40"
path1="F:\\lpw\\ModelNet_Blender_OFF2Multiview-master\\40train"
list=os.listdir(path)

for i in list:
    fpath=path+"\\"+i
    if i!="airplane":
        if os.path.isdir(fpath):
            fpath=fpath+"\\train"
            listoff=os.listdir(fpath)
            for j in listoff:
                if(j[len(j)-3:]=="off"):
                    print(fpath + "\\" + j)
                    shutil.copyfile(fpath+"\\"+j, path1+"\\"+i+"\\"+j)
