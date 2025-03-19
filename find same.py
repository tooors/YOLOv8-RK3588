import os

def remove_duplicate_files(folder1, folder2):
    # 获取第二个文件夹中的所有文件名
    files_in_folder2 = set(os.listdir(folder2))

    # 遍历第一个文件夹中的所有文件
    for filename in os.listdir(folder1):
        file_path = os.path.join(folder1, filename)
        # 检查文件是否存在于第二个文件夹中
        if filename in files_in_folder2 and os.path.isfile(file_path):
            # 移除第一个文件夹中的相同文件
            os.remove(file_path)
            print(f"已移除文件: {file_path}")

# 示例用法
folder1 = r"C:\Users\21853\Desktop\25GCS_v5_串行增强\2332246_1741846040\yolo_labels"
folder2 = r"C:\Users\21853\Desktop\harmful2\yolo_labels"
remove_duplicate_files(folder1, folder2)