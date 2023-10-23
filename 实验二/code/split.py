import os
import shutil

# 定义文件夹路径
image_folder='/home/ntire23_2/mwz/ultralytics-main/data/images'
files=os.listdir(image_folder)
l=len(files)
train=files[0:int(0.8*l)]
valid=files[int(0.8*l):int(0.9*l)]
test=files[int(0.9*l):]

f=open('data/dataset/train.txt','w')
for file in train:
    file_name=os.path.join(image_folder,file)
    f.write(file_name+'\n')
f.close()
f=open('data/dataset/valid.txt','w')
for file in valid:
    file_name=os.path.join(image_folder,file)
    f.write(file_name+'\n')
f.close()
f=open('data/dataset/test.txt','w')
for file in test:
    file_name=os.path.join(image_folder,file)
    f.write(file_name+'\n')
f.close()