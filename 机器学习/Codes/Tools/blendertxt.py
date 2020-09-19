import os

txtpath="F:\\lpw\\ModelNet_Blender_OFF2Multiview-master\\datatxt\\test_data\\test10.txt"
f=open(txtpath,'w')

path="F:\\lpw\\ModelNet_Blender_OFF2Multiview-master\\10test"
list=os.listdir(path)

for i in list:
    fpath=path+"\\"+i
    dirlist=os.listdir(fpath)

    for j in dirlist:
        print(j)
        str=".\\Air\\"+i+"\\"+j+"\n"
        f.write(str)
