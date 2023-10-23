import os
import shutil

# 定义文件夹路径
label_folder = "SICISP2023\\2-OBJECTDETECTION\\labels"
image_folder = "SICISP2023\\2-OBJECTDETECTION\\images"
output_folder1 = "SICISP2023\\2-OBJECTDETECTION\\image_select"
output_folder2 = "SICISP2023\\2-OBJECTDETECTION\\label_select"

# 获取label文件夹中的所有文件名（去除扩展名）
label_files = [os.path.splitext(file)[0] for file in os.listdir(label_folder) if file.endswith(".txt")]\

# 遍历image文件夹中的文件
for file in os.listdir(image_folder):
    if file.endswith(".jpg"):
        # 获取当前文件的文件名（去除扩展名）
        file_name = os.path.splitext(file)[0]
        
        # 如果当前文件名存在于label_files列表中
        if file_name in label_files:
            # 构建源文件和目标文件的路径
            source_path = os.path.join(image_folder, file)
            destination_path = os.path.join(output_folder1, file)
            
            # 复制文件到新的文件夹中
            shutil.copyfile(source_path, destination_path)
            print(f"复制文件: {file}")

# 获取images文件夹中的所有文件名（去除扩展名）
image_files = [os.path.splitext(file)[0] for file in os.listdir(output_folder1) if file.endswith(".jpg")]
# 遍历label文件夹中的文件
for file in os.listdir(label_folder):
    if file.endswith(".txt"):
        # 获取当前文件的文件名（去除扩展名）
        file_name = os.path.splitext(file)[0]
        
        # 如果当前文件名存在于label_files列表中
        if file_name in image_files:
            # 构建源文件和目标文件的路径
            source_path = os.path.join(label_folder, file)
            destination_path = os.path.join(output_folder2, file)
            
            # 复制文件到新的文件夹中
            shutil.copyfile(source_path, destination_path)
            print(f"复制文件: {file}")