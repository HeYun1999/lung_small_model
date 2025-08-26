import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import logging
import albumentations as A
from albumentations.pytorch import ToTensorV2
from monai.networks.nets import UNet
from monai.metrics import DiceMetric
from sklearn.metrics import recall_score, f1_score
import time

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置随机种子，保证结果可复现
np.random.seed(42)
torch.manual_seed(42)

def setup_device():
    """设置训练设备（GPU或CPU）"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f'使用GPU设备: {torch.cuda.get_device_name(0)}')
    else:
        device = torch.device('cpu')
        logger.info('使用CPU设备')
    return device

class LungDataset(Dataset):
    """肺部图像分割数据集"""
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        
        # 获取所有图像文件名（不包括扩展名）
        self.image_filenames = [f.split('.')[0] for f in os.listdir(image_dir) if f.endswith('.png')]
        self.label_filenames = [f.split('.')[0] for f in os.listdir(label_dir) if f.endswith('.png')]
        
        # 确保图像和标签文件匹配
        self.filenames = list(set(self.image_filenames) & set(self.label_filenames))
        self.filenames.sort()  # 保持顺序一致
        
        if len(self.filenames) == 0:
            raise ValueError(f'未找到匹配的图像和标签文件在 {image_dir} 和 {label_dir}')
        
        logger.info(f'加载了 {len(self.filenames)} 个样本')
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image_path = os.path.join(self.image_dir, f'{filename}.png')
        label_path = os.path.join(self.label_dir, f'{filename}.png')
        
        # 加载图像和标签
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('L')  # 转为灰度图
        
        # 转换为numpy数组
        image = np.array(image)
        label = np.array(label)
        
        # 修复：根据用户提供的信息，将灰度值0、100、255映射到类别0、1、2
        # 创建映射字典
        label_mapping = {
            0: 0,    # 第一类保持不变
            100: 1,  # 第二类映射到1
            255: 2   # 第三类映射到2
        }
        
        # 创建一个新的标签数组
        mapped_label = np.zeros_like(label, dtype=np.int64)
        
        # 应用映射
        for gray_value, class_id in label_mapping.items():
            mapped_label[label == gray_value] = class_id
        
        # 确保所有值都在有效范围内
        mapped_label = np.clip(mapped_label, 0, 2)
        
        # 应用数据增强
        if self.transform:
            augmented = self.transform(image=image, mask=mapped_label)
            image = augmented['image']
            mapped_label = augmented['mask']
        
        return image, mapped_label

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        # 修复：将np.Inf改为np.inf以兼容NumPy 2.0
        self.val_loss_min = np.inf
        self.delta = delta
    
    def __call__(self, val_loss, model, model_path):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logger.info(f'早停计数器: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_path)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model, model_path):
        if self.verbose:
            logger.info(f'验证损失减少 ({self.val_loss_min:.6f} --> {val_loss:.6f}). 保存模型...')
        torch.save(model.state_dict(), model_path)
        self.val_loss_min = val_loss

def compute_iou(pred, target, num_classes=3):
    """计算IoU（交集除以并集）"""
    # 创建混淆矩阵
    confusion_matrix = torch.zeros(num_classes, num_classes, device=pred.device)
    for p, t in zip(pred.flatten(), target.flatten()):
        confusion_matrix[p, t] += 1
    
    # 计算每个类别的IoU
    iou_list = []
    for i in range(num_classes):
        true_positive = confusion_matrix[i, i].item()
        false_positive = confusion_matrix[i, :].sum().item() - true_positive
        false_negative = confusion_matrix[:, i].sum().item() - true_positive
        
        # 避免除以0
        if true_positive + false_positive + false_negative == 0:
            iou = 0.0
        else:
            iou = true_positive / (true_positive + false_positive + false_negative)
        
        iou_list.append(iou)
    
    return iou_list

def train_epoch(model, loader, loss_function, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    start_time = time.time()
    total_batches = len(loader)
    
    for i, (images, labels) in enumerate(loader):
        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)
        
        # 前向传播
        outputs = model(images)
        loss = loss_function(outputs, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 累计损失
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        # 打印批次进度
        if (i + 1) % 5 == 0 or (i + 1) == total_batches:
            logger.info(f'  训练批次 {i+1}/{total_batches}, 批次损失: {loss.item():.4f}')
    
    avg_loss = total_loss / total_samples
    logger.info(f'训练损失: {avg_loss:.4f}, 耗时: {time.time() - start_time:.2f}秒')
    
    return avg_loss

def val_epoch(model, loader, loss_function, device):
    """验证一个epoch"""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    all_outputs = []
    all_labels = []
    
    start_time = time.time()
    total_batches = len(loader)
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            
            # 前向传播
            outputs = model(images)
            loss = loss_function(outputs, labels)
            
            # 累计损失
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # 保存输出和标签用于计算指标
            all_outputs.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            # 打印批次进度
            if (i + 1) % 5 == 0 or (i + 1) == total_batches:
                logger.info(f'  验证批次 {i+1}/{total_batches}, 批次损失: {loss.item():.4f}')
    
    # 计算指标
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # 转换为torch张量
    all_outputs_tensor = torch.tensor(all_outputs, device=device)
    all_labels_tensor = torch.tensor(all_labels, device=device)
    
    # 计算预测值
    _, predictions = torch.max(all_outputs_tensor, dim=1)
    
    # 计算每个类别的IoU
    iou_scores = compute_iou(predictions, all_labels_tensor, num_classes=3)
    miou = np.mean(iou_scores)
    
    # 打印每个类别的IoU值
    for i, iou in enumerate(iou_scores):
        logger.info(f'类别{i} IoU: {iou:.4f}')
    logger.info(f'mIoU: {miou:.4f}')
    
    # 计算准确率
    accuracy = np.mean(predictions.cpu().numpy() == all_labels)
    logger.info(f'准确率: {accuracy:.4f}')
    
    # 计算Recall和F1分数
    recall = recall_score(all_labels.flatten(), predictions.cpu().numpy().flatten(), average=None, labels=[0, 1, 2], zero_division=0)
    f1 = f1_score(all_labels.flatten(), predictions.cpu().numpy().flatten(), average=None, labels=[0, 1, 2], zero_division=0)
    
    mean_recall = np.mean(recall)
    mean_f1 = np.mean(f1)
    
    logger.info(f'类别0 Recall: {recall[0]:.4f}, 类别1 Recall: {recall[1]:.4f}, 类别2 Recall: {recall[2]:.4f}')
    logger.info(f'类别0 F1: {f1[0]:.4f}, 类别1 F1: {f1[1]:.4f}, 类别2 F1: {f1[2]:.4f}')
    logger.info(f'平均Recall: {mean_recall:.4f}, 平均F1: {mean_f1:.4f}')
    
    avg_loss = total_loss / total_samples
    logger.info(f'验证损失: {avg_loss:.4f}, 耗时: {time.time() - start_time:.2f}秒')
    
    return avg_loss, miou, mean_recall, mean_f1

def visualize_results(model, loader, device, save_dir='results'):
    """可视化模型预测结果"""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        # 只可视化前3个样本
        for i, (images, labels) in enumerate(loader):
            if i >= 3:
                break
            
            images = images.to(device, dtype=torch.float32)
            outputs = model(images)
            _, predictions = torch.max(outputs, dim=1)
            
            # 转换为numpy数组
            image = images[0].cpu().permute(1, 2, 0).numpy()
            label = labels[0].cpu().numpy()
            pred = predictions[0].cpu().numpy()
            
            # 归一化图像到[0, 1]
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
            
            # 创建子图
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(image)
            axes[0].set_title('输入图像')
            axes[0].axis('off')
            
            axes[1].imshow(label, cmap='jet')
            axes[1].set_title('真实标签')
            axes[1].axis('off')
            
            axes[2].imshow(pred, cmap='jet')
            axes[2].set_title('预测结果')
            axes[2].axis('off')
            
            # 保存图像
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'prediction_{i}.png'))
            plt.close()
    
    logger.info(f'预测结果已保存到 {save_dir} 目录')

def plot_metrics(train_losses, val_losses, val_mious, val_recalls, val_f1s):
    """绘制训练和验证指标曲线"""
    plt.figure(figsize=(15, 10))
    
    # 绘制损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.title('损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    
    # 绘制mIoU曲线
    plt.subplot(2, 2, 2)
    plt.plot(val_mious)
    plt.title('mIoU曲线')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.grid(True)
    
    # 绘制Recall曲线
    plt.subplot(2, 2, 3)
    plt.plot(val_recalls)
    plt.title('Recall曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.grid(True)
    
    # 绘制F1曲线
    plt.subplot(2, 2, 4)
    plt.plot(val_f1s)
    plt.title('F1曲线')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('metrics.png')
    plt.close()
    
    logger.info('指标曲线已保存为 metrics.png')

def main():
    # 设置设备
    device = setup_device()
    
    # 设置数据集路径
    data_dir = '/mnt/sda2/heyun_data/lung2_data'
    train_image_dir = os.path.join(data_dir, 'train', 'image')
    train_label_dir = os.path.join(data_dir, 'train', 'label')
    val_image_dir = os.path.join(data_dir, 'test', 'image')
    val_label_dir = os.path.join(data_dir, 'test', 'label')
    
    # 检查数据集目录是否存在
    for dir_path in [train_image_dir, train_label_dir, val_image_dir, val_label_dir]:
        if not os.path.exists(dir_path):
            logger.error(f'数据集目录不存在: {dir_path}')
            raise FileNotFoundError(f'数据集目录不存在: {dir_path}')
    
    # 数据增强和转换
    train_transform = A.Compose([
        A.Resize(height=256, width=256),
        A.RandomRotate90(p=0.5),
        # 修复：将A.Flip替换为A.HorizontalFlip和A.VerticalFlip
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(height=256, width=256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    # 创建数据集和数据加载器
    batch_size = 4
    train_dataset = LungDataset(train_image_dir, train_label_dir, transform=train_transform)
    val_dataset = LungDataset(val_image_dir, val_label_dir, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 检查数据集大小
    if len(train_dataset) < 10:
        logger.warning(f'训练集样本数量较少 ({len(train_dataset)}), 可能影响模型性能')
    if len(val_dataset) < 5:
        logger.warning(f'验证集样本数量较少 ({len(val_dataset)}), 评估结果可能不可靠')
    
    # 创建模型
    model = UNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2
    ).to(device)
    
    # 设置损失函数和优化器
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # 设置早停机制
    early_stopping = EarlyStopping(patience=15, verbose=True)
    
    # 创建模型保存目录
    os.makedirs('models', exist_ok=True)
    model_path = 'models/best_model.pth'
    
    # 训练循环
    num_epochs = 300
    train_losses = []
    val_losses = []
    val_mious = []
    val_recalls = []
    val_f1s = []
    
    logger.info(f'开始训练模型，共 {num_epochs} 个epoch')
    
    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch+1}/{num_epochs}')
        
        # 训练一个epoch
        train_loss = train_epoch(model, train_loader, loss_function, optimizer, device)
        
        # 验证一个epoch
        val_loss, val_miou, val_recall, val_f1 = val_epoch(model, val_loader, loss_function, device)
        
        # 记录指标
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_mious.append(val_miou)
        val_recalls.append(val_recall)
        val_f1s.append(val_f1)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 早停检查
        early_stopping(val_loss, model, model_path)
        if early_stopping.early_stop:
            logger.info('触发早停机制，停止训练')
            break
        
        # 每10个epoch保存一次模型
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'models/model_epoch_{epoch+1}.pth')
            logger.info(f'模型已保存到 models/model_epoch_{epoch+1}.pth')
    
    # 绘制指标曲线
    plot_metrics(train_losses, val_losses, val_mious, val_recalls, val_f1s)
    
    # 加载最佳模型
    model.load_state_dict(torch.load(model_path))
    logger.info(f'加载最佳模型: {model_path}')
    
    # 可视化结果
    visualize_results(model, val_loader, device)
    
    logger.info('训练完成！')

if __name__ == '__main__':
    main()