import os
import shutil

def organize_dataset(datasets_dir, output_dir, dataset_name):
    """
    整理数据集到指定格式
    
    参数:
    images_dir: 原始图像文件夹路径
    output_dir: 输出目录路径
    dataset_name: 数据集名称
    """
    
    # 创建目标目录结构
    dataset_path = os.path.join(output_dir, dataset_name)
    train_images_dir = os.path.join(dataset_path, 'train', 'images')
    train_masks_dir = os.path.join(dataset_path, 'train', 'masks')
    test_images_dir = os.path.join(dataset_path, 'test', 'images')
    test_masks_dir = os.path.join(dataset_path, 'test', 'masks')
    
    # 创建所有必要的目录
    for dir_path in [train_images_dir, train_masks_dir, test_images_dir, test_masks_dir]:
        os.makedirs(dir_path, exist_ok=True)

    dataset_dir = os.path.join(datasets_dir, dataset_name)
    
    images_dir = os.path.join(dataset_dir, 'images')
    masks_dir = os.path.join(dataset_dir, 'masks')
    idx_dir = os.path.join(dataset_dir, 'img_idx')
    # 读取索引文件
    train_txt = os.path.join(idx_dir, 'train_'+ dataset_name +'.txt')
    test_txt = os.path.join(idx_dir, 'test_' + dataset_name +'.txt')

    print(train_txt)
    print(test_txt)
    
    # 处理训练集
    if os.path.exists(train_txt):
        with open(train_txt, 'r') as f:
            train_files = [line.strip() for line in f.readlines()]

        
        for file_name in train_files:
            # print(file_name)
            # 复制图像
            src_image = os.path.join(images_dir, file_name + '.png')
            dst_image = os.path.join(train_images_dir, file_name + '.png')
            if os.path.exists(src_image):
                shutil.copy2(src_image, dst_image)
                print(f"Copied {src_image} to {dst_image}")
            
            # 复制掩码
            src_mask = os.path.join(masks_dir, file_name + '.png')
            dst_mask = os.path.join(train_masks_dir, file_name + '.png')
            if os.path.exists(src_mask):
                shutil.copy2(src_mask, dst_mask)
    
    # 处理测试集
    if os.path.exists(test_txt):
        with open(test_txt, 'r') as f:
            test_files = [line.strip() for line in f.readlines()]
        
        for file_name in test_files:
            # 复制图像
            src_image = os.path.join(images_dir, file_name + '.png')
            dst_image = os.path.join(test_images_dir, file_name + '.png')
            if os.path.exists(src_image):
                shutil.copy2(src_image, dst_image)
            
            # 复制掩码
            src_mask = os.path.join(masks_dir, file_name + '.png')
            dst_mask = os.path.join(test_masks_dir, file_name + '.png')
            if os.path.exists(src_mask):
                shutil.copy2(src_mask, dst_mask)
    
    print(f"数据集已成功整理到: {dataset_path}")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='整理数据集格式')
    # parser.add_argument('--images_dir', type=str, default='', help='原始图像文件夹路径')
    # parser.add_argument('--masks_dir', type=str, required=True, help='原始掩码文件夹路径')
    # parser.add_argument('--idx_dir', type=str, required=True, help='包含txt文件的文件夹路径')
    # parser.add_argument('--output_dir', type=str, default='./datasets', help='输出目录路径')
    # parser.add_argument('--dataset_name', type=str, required=True, help='数据集名称')
    
    # args = parser.parse_args()
    
    # organize_dataset(datasets_dir='/mnt/d/code/IRSTD/datasets', output_dir='/mnt/d/code/IRSTD/datasetsV2', dataset_name='NUDT-SIRST')
    # organize_dataset(datasets_dir='/mnt/d/code/IRSTD/datasets', output_dir='/mnt/d/code/IRSTD/datasetsV2', dataset_name='NUAA-SIRST')
    organize_dataset(datasets_dir='/mnt/d/code/IRSTD/datasets', output_dir='/mnt/d/code/IRSTD/datasetsV2', dataset_name='IRSTD-1K')