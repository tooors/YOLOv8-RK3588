import os


def replace_first_parameter(folder_path):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # 只处理文件
        if os.path.isfile(file_path):
            # 读取文件内容
            with open(file_path, 'r') as f:
                lines = f.readlines()

            # 修改内容：将每行的第一个参数 0 改为 3
            modified_lines = []
            for line in lines:
                parts = line.strip().split()
                if parts and parts[0] == '0':
                    parts[0] = '2'
                    modified_line = ' '.join(parts) + '\n'
                    modified_lines.append(modified_line)
                else:
                    modified_lines.append(line)

            # 将修改后的内容写回文件
            with open(file_path, 'w') as f:
                f.writelines(modified_lines)

            print(f"已修改文件: {filename}")


# 示例用法
folder_path = r"C:\Users\21853\Desktop\harmful\yolo_labels"
replace_first_parameter(folder_path)