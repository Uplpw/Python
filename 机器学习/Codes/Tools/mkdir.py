import os
import random

# path="F:\\lpw\\Tranfer\\fat2fit\\test"
path="F:\\lpw\\Tranfer\\finger2pants_end\\test"
list=os.listdir(path)

for i in range(40):
    fpath=path+"\\"+str(i+1001)
    if(os.path.exists(fpath)):
        listdir=os.listdir(fpath)
        for j in listdir:
            if(j=="AtoB"):
                ffpath=fpath+"\\"+j
                Apng=ffpath+"\\Apng"
                Bpng = ffpath + "\\Bpng"
                if(os.path.exists(Apng)):
                    pass
                else:
                    os.mkdir(Apng)
                if (os.path.exists(Bpng)):
                    pass
                else:
                    os.mkdir(Bpng)
            if (j == "BtoA"):
                ffpath = fpath + "\\" + j
                Apng = ffpath + "\\Apng"
                Bpng = ffpath + "\\Bpng"
                if (os.path.exists(Apng)):
                    pass
                else:
                    os.mkdir(Apng)
                if (os.path.exists(Bpng)):
                    pass
                else:
                    os.mkdir(Bpng)
    else:
        os.mkdir(fpath)