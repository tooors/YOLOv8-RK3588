import json
import os

def convert_coco_to_yolo(coco_json_path, output_dir):
    # 读取 COCO JSON 文件
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取类别信息
    categories = {category['id']: category['name'] for category in coco_data['categories']}

    # 遍历每个图像
    for image_info in coco_data['images']:
        image_id = image_info['id']
        image_width = image_info['width']
        image_height = image_info['height']
        image_filename = image_info['file_name']

        # 获取该图像的所有标注
        annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]

        # 如果没有标注，跳过
        if not annotations:
            continue

        # 创建 YOLO 格式的标注文件
        txt_filename = os.path.splitext(image_filename)[0] + '.txt'
        txt_path = os.path.join(output_dir, txt_filename)

        with open(txt_path, 'w') as f:
            for ann in annotations:
                # 获取类别 ID 和边界框信息
                category_id = ann['category_id']
                bbox = ann['bbox']  # [x_min, y_min, width, height]

                # 转换为 YOLO 格式 (x_center, y_center, width, height)
                x_min, y_min, width, height = bbox
                x_center = (x_min + width / 2) / image_width
                y_center = (y_min + height / 2) / image_height
                yolo_width = width / image_width
                yolo_height = height / image_height

                # 写入 TXT 文件
                f.write(f"{category_id} {x_center:.6f} {y_center:.6f} {yolo_width:.6f} {yolo_height:.6f}\n")

    print(f"转换完成，YOLO 格式的标注文件已保存到 {output_dir}")

# 示例用法
coco_json_path = r"C:\Users\21853\Desktop\25GCS_v5_并行增强\2332255_1741849467\Annotations\coco_info.json"
output_dir = r"C:\Users\21853\Desktop\25GCS_v5_并行增强\2332255_1741849467\Annotations\yolo_labels"

convert_coco_to_yolo(coco_json_path, output_dir)