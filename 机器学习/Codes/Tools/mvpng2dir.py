import os
import shutil
path="F:\\lpw\\Tranfer\\finger2pants\\final2_duiqi"
pngpath="F:\\lpw\\Tranfer\\finger2pants\\new"

list=os.listdir(path)
for i in list:
    fpath=path+"\\"+i
    length = len(i)
    if(i[length-3:]=="obj"):
        print(i)
        shutil.copyfile(fpath, pngpath+"\\"+i)