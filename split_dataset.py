import os
import numpy as np
import shutil
import logging
from sklearn.model_selection import train_test_split
from PIL import Image
import random

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置随机种子，保证结果可复现
np.random.seed(42)
random.seed(42)

def split_dataset():
    # 数据集根目录
    data_dir = '/mnt/sda2/heyun_data/lung2_data'
    
    # 图像和标签目录
    image_dir = os.path.join(data_dir, 'image')
    label_dir = os.path.join(data_dir, 'label')
    
    # 检查数据集目录是否存在
    if not os.path.exists(data_dir):
        logger.error(f'数据集目录不存在: {data_dir}')
        raise FileNotFoundError(f'数据集目录不存在: {data_dir}')
    
    if not os.path.exists(image_dir):
        logger.error(f'图像目录不存在: {image_dir}')
        raise FileNotFoundError(f'图像目录不存在: {image_dir}')
    
    if not os.path.exists(label_dir):
        logger.error(f'标签目录不存在: {label_dir}')
        raise FileNotFoundError(f'标签目录不存在: {label_dir}')
    
    # 创建训练集和测试集目录
    train_image_dir = os.path.join(data_dir, 'train', 'image')
    train_label_dir = os.path.join(data_dir, 'train', 'label')
    test_image_dir = os.path.join(data_dir, 'test', 'image')
    test_label_dir = os.path.join(data_dir, 'test', 'label')
    
    # 创建目录结构
    for dir_path in [train_image_dir, train_label_dir, test_image_dir, test_label_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # 加载图像和对应的标签
    logger.info('开始加载数据集...')
    image_paths = []
    mask_paths = []
    
    for file in os.listdir(image_dir):
        if file.lower().endswith('.png'):
            # 提取文件名（不含扩展名）
            file_name = os.path.splitext(file)[0]
            image_path = os.path.join(image_dir, file)
            mask_path = os.path.join(label_dir, f'{file_name}.png')
            if os.path.exists(mask_path):
                # 验证图像和标签是否有效
                try:
                    img = Image.open(image_path)
                    mask = Image.open(mask_path)
                    img.verify()
                    mask.verify()
                    image_paths.append(image_path)
                    mask_paths.append(mask_path)
                except Exception as e:
                    logger.warning(f'文件验证失败: {file_name}.png, 错误: {str(e)}, 跳过此图像')
            else:
                logger.warning(f'未找到标签文件: {mask_path}，跳过此图像')
    
    logger.info(f'成功加载 {len(image_paths)} 对图像和标签')
    
    # 检查是否有足够的数据
    if len(image_paths) < 10:
        logger.warning(f'数据集较小，只有 {len(image_paths)} 对样本，可能会影响模型性能')
    
    # 划分训练集和测试集 (8:2)
    train_images, test_images, train_masks, test_masks = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )
    
    logger.info(f'训练集样本数: {len(train_images)}')
    logger.info(f'测试集样本数: {len(test_images)}')
    
    # 复制训练集文件
    logger.info('开始复制训练集文件...')
    copy_files(train_images, train_image_dir)
    copy_files(train_masks, train_label_dir)
    
    # 复制测试集文件
    logger.info('开始复制测试集文件...')
    copy_files(test_images, test_image_dir)
    copy_files(test_masks, test_label_dir)
    
    # 保存划分结果的索引文件
    save_split_info(train_images, test_images, data_dir)
    
    logger.info('数据集划分完成！')
    logger.info(f'训练集图像: {train_image_dir}')
    logger.info(f'训练集标签: {train_label_dir}')
    logger.info(f'测试集图像: {test_image_dir}')
    logger.info(f'测试集标签: {test_label_dir}')

def copy_files(file_paths, target_dir):
    """复制文件到目标目录"""
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        target_path = os.path.join(target_dir, file_name)
        try:
            shutil.copy2(file_path, target_path)
        except Exception as e:
            logger.error(f'复制文件失败: {file_path} -> {target_path}, 错误: {str(e)}')

def save_split_info(train_images, test_images, data_dir):
    """保存划分信息到文本文件"""
    # 保存训练集和测试集的文件名
    train_files = [os.path.basename(f) for f in train_images]
    test_files = [os.path.basename(f) for f in test_images]
    
    # 保存训练集文件列表
    with open(os.path.join(data_dir, 'train_files.txt'), 'w') as f:
        for file in train_files:
            f.write(f'{file}\n')
    
    # 保存测试集文件列表
    with open(os.path.join(data_dir, 'test_files.txt'), 'w') as f:
        for file in test_files:
            f.write(f'{file}\n')
    
    # 保存划分统计信息
    with open(os.path.join(data_dir, 'split_info.txt'), 'w') as f:
        f.write(f'总样本数: {len(train_images) + len(test_images)}\n')
        f.write(f'训练集样本数: {len(train_images)} ({len(train_images) / (len(train_images) + len(test_images)) * 100:.1f}%)\n')
        f.write(f'测试集样本数: {len(test_images)} ({len(test_images) / (len(train_images) + len(test_images)) * 100:.1f}%)\n')
        f.write(f'随机种子: 42\n')

if __name__ == '__main__':
    split_dataset()