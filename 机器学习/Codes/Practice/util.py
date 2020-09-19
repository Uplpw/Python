##-------------utils---------------##

import os
import shutil

def logdirs(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
    list=os.listdir(path)
    if(len(list)==0):
        pass
    else:
        for i in list:
            fpath=path+"\\"+i
            print(fpath)
            if(os.path.isdir(fpath)):
                shutil.rmtree(fpath)
            if(os.path.isfile(fpath)):
                os.remove(fpath)