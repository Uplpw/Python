import os
import shutil
path="F:\\software\\Matlab2019"
list=os.listdir(path)
for i in list:
    print(i)
    dir=path+"\\"+i
    length=len(i)
    if os.path.isdir(dir):
        print(dir)
        shutil.rmtree(dir)
    else:
        os.remove(dir)
