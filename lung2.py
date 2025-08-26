import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import monai
from monai.networks.nets import UNet
from monai.metrics import DiceMetric, compute_iou
from monai.metrics.utils import get_mask_edges, get_surface_distance
import logging
import os
import re

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置随机种子，保证结果可复现
np.random.seed(42)
torch.manual_seed(42)

# 修改数据目录定义，指向已划分好的数据集
data_dir = '/mnt/sda2/heyun_data/lung2_data'

# 训练集和验证集的图像和标签目录
train_image_dir = os.path.join(data_dir, 'train', 'image')
train_label_dir = os.path.join(data_dir, 'train', 'label')
val_image_dir = os.path.join(data_dir, 'test', 'image')
val_label_dir = os.path.join(data_dir, 'test', 'label')

# 检查数据集目录是否存在
if not os.path.exists(data_dir):
    logger.error(f'数据集目录不存在: {data_dir}')
    raise FileNotFoundError(f'数据集目录不存在: {data_dir}')

if not os.path.exists(train_image_dir):
    logger.error(f'训练集图像目录不存在: {train_image_dir}')
    raise FileNotFoundError(f'训练集图像目录不存在: {train_image_dir}')

if not os.path.exists(train_label_dir):
    logger.error(f'训练集标签目录不存在: {train_label_dir}')
    raise FileNotFoundError(f'训练集标签目录不存在: {train_label_dir}')

if not os.path.exists(val_image_dir):
    logger.error(f'验证集图像目录不存在: {val_image_dir}')
    raise FileNotFoundError(f'验证集图像目录不存在: {val_image_dir}')

if not os.path.exists(val_label_dir):
    logger.error(f'验证集标签目录不存在: {val_label_dir}')
    raise FileNotFoundError(f'验证集标签目录不存在: {val_label_dir}')

# 加载训练集图像和对应的标签
logger.info('开始加载训练集...')
train_images = []
train_masks = []

for file in os.listdir(train_image_dir):
    if file.lower().endswith('.png'):
        # 提取文件名（不含扩展名）
        file_name = os.path.splitext(file)[0]
        image_path = os.path.join(train_image_dir, file)
        mask_path = os.path.join(train_label_dir, f'{file_name}.png')
        if os.path.exists(mask_path):
            train_images.append(image_path)
            train_masks.append(mask_path)
        else:
            logger.warning(f'未找到训练集标签文件: {mask_path}，跳过此图像')

logger.info(f'成功加载 {len(train_images)} 对训练集图像和标签')

# 加载验证集图像和对应的标签
logger.info('开始加载验证集...')
val_images = []
val_masks = []

for file in os.listdir(val_image_dir):
    if file.lower().endswith('.png'):
        # 提取文件名（不含扩展名）
        file_name = os.path.splitext(file)[0]
        image_path = os.path.join(val_image_dir, file)
        mask_path = os.path.join(val_label_dir, f'{file_name}.png')
        if os.path.exists(mask_path):
            val_images.append(image_path)
            val_masks.append(mask_path)
        else:
            logger.warning(f'未找到验证集标签文件: {mask_path}，跳过此图像')

logger.info(f'成功加载 {len(val_images)} 对验证集图像和标签')

# 检查是否有足够的数据
if len(val_images) < 5:
    logger.warning(f'验证集较小，只有 {len(val_images)} 对样本，可能会影响模型评估的准确性')

# 删除重复的数据加载和划分代码
# 数据增强和预处理
# 删除原有的idx == 0条件，改为在初始化时分析整个数据集
class LungDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform
        
        # 分析整个数据集的类别分布
        self.analyze_dataset()
        
    def analyze_dataset(self):
        # 只分析前20个样本以节省时间
        num_samples = min(20, len(self.images))
        all_classes = []
        
        for i in range(num_samples):
            original_mask = np.array(Image.open(self.masks[i]).convert('L'))
            mask = np.zeros_like(original_mask, dtype=np.float32)
            mask[original_mask == 100] = 1
            mask[original_mask == 255] = 2
            
            all_classes.extend(np.unique(mask).tolist())
            
        unique_classes = np.unique(all_classes)
        logger.info(f'数据集中出现的类别: {unique_classes}')
        
        if 1 in unique_classes:
            logger.info('类别1存在于数据集中')
        else:
            logger.info('警告：在前{num_samples}个样本中未发现类别1')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.images[idx]).convert('RGB'))
        
        # 加载标签图像并转换为灰度
        original_mask = np.array(Image.open(self.masks[idx]).convert('L'))
        
        # 标签处理：将0、100、255转换为0、1、2三个类别
        mask = np.zeros_like(original_mask, dtype=np.float32)
        mask[original_mask == 100] = 1  # 轻度病变
        mask[original_mask == 255] = 2  # 重度病变
        # 背景保持为0
        
        # 调试信息：检查每个样本的标签分布（每100个样本打印一次）
        if idx % 100 == 0:
            logger.info(f'样本 {idx} 标签唯一值: {np.unique(mask)}')
            logger.info(f'样本 {idx} 类别0比例: {np.sum(mask == 0) / mask.size:.4f}')
            logger.info(f'样本 {idx} 类别1比例: {np.sum(mask == 1) / mask.size:.4f}')
            logger.info(f'样本 {idx} 类别2比例: {np.sum(mask == 2) / mask.size:.4f}')

        if self.transform:
            # 直接使用2D掩码，不需要squeeze
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            # 变换后添加通道维度
            mask = mask[np.newaxis, ...]

        return image, mask

# 改进的数据增强变换
# 简化的数据增强策略，移除可能导致过度失真的操作
train_transform = A.Compose([
    A.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.7),
    A.Resize(height=512, width=512),
    A.Rotate(limit=30, p=0.5),  # 减少旋转角度
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),  # 降低概率
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(height=512, width=512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# 创建数据集和数据加载器
train_dataset = LungDataset(train_images, train_masks, transform=train_transform)
val_dataset = LungDataset(val_images, val_masks, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

# 定义模型
def create_model(num_classes=3):  # 修改为默认3分类
    # 使用Monai的UNet模型
    model = UNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=num_classes,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
    return model

# 初始化模型、损失函数和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'使用设备: {device}')

model = create_model(num_classes=3).to(device)  # 明确指定3分类
# 导入额外的损失函数
import torch.nn as nn
from monai.losses import DiceLoss

# 使用混合损失函数 (CrossEntropyLoss + DiceLoss)
class MixedLoss(nn.Module):
    def __init__(self, ce_weight=0.5, dice_weight=0.5):
        super(MixedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()  # 从torch.nn导入
        self.dice_loss = DiceLoss(softmax=True)  # 多分类使用softmax
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        # CrossEntropyLoss需要targets是long类型，且没有通道维度
        ce_loss = self.ce_loss(inputs, targets.long().squeeze(1))
        
        # DiceLoss需要targets与inputs有相同的通道数
        # 我们需要将targets转换为one-hot编码格式
        from monai.networks.utils import one_hot
        targets_onehot = one_hot(targets.long(), num_classes=3)
        
        dice_loss = self.dice_loss(inputs, targets_onehot)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss

loss_function = MixedLoss(ce_weight=0.5, dice_weight=0.5)

# 使用AdamW优化器，调整学习率
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)

# 使用更灵活的学习率调度器
from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True, min_lr=1e-6)
# 移除MONAI的DiceMetric，使用自定义实现
# metric = DiceMetric(include_background=True, reduction='mean')

# 创建模型保存目录
model_dir = 'models_lung7'

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    logger.info(f'创建模型保存目录: {model_dir}')

# 训练函数
def train_epoch(model, loader, optimizer, loss_function, device):
    model.train()
    total_loss = 0
    total_batches = len(loader)

    for i, batch_data in enumerate(loader):
        inputs, labels = batch_data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)

        # 打印批次进度
        if (i + 1) % 5 == 0 or (i + 1) == total_batches:
            logger.info(f'  训练批次 {i+1}/{total_batches}, 批次损失: {loss.item():.4f}')

    return total_loss / len(loader.dataset)

# 验证函数 - 改进为多分类评估
def val_epoch(model, loader, loss_function, device):
    model.eval()
    total_loss = 0
    total_batches = len(loader)
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for i, batch_data in enumerate(loader):
            inputs, labels = batch_data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            total_loss += loss.item() * inputs.size(0)

            # 保存输出和标签用于后续指标计算
            all_outputs.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            # 打印批次进度
            if (i + 1) % 5 == 0 or (i + 1) == total_batches:
                logger.info(f'  验证批次 {i+1}/{total_batches}, 批次损失: {loss.item():.4f}')

    # 计算指标
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_labels = all_labels.squeeze(1)  # 移除通道维度

    # 新增：检查验证集整体类别分布
    logger.info(f'验证集标签唯一值: {np.unique(all_labels)}')
    logger.info(f'验证集类别0比例: {np.sum(all_labels == 0) / all_labels.size:.4f}')
    logger.info(f'验证集类别1比例: {np.sum(all_labels == 1) / all_labels.size:.4f}')
    logger.info(f'验证集类别2比例: {np.sum(all_labels == 2) / all_labels.size:.4f}')

    # 转换为one-hot编码以便计算Dice
    from monai.networks.utils import one_hot
    all_outputs_softmax = torch.softmax(torch.tensor(all_outputs), dim=1)
    all_outputs_onehot = one_hot(torch.argmax(all_outputs_softmax, dim=1).unsqueeze(1), num_classes=3)
    all_labels_onehot = one_hot(torch.tensor(all_labels).unsqueeze(1), num_classes=3)

    # 计算每个类别的Dice
    dice_metric = DiceMetric(include_background=True, reduction='none')
    dice_scores = dice_metric(all_outputs_onehot, all_labels_onehot)
    
    # 计算每个类别的平均Dice分数
    mean_dice_per_class = dice_scores.mean(dim=0)
    
    # 新增：处理nan值
    mean_dice_per_class = torch.nan_to_num(mean_dice_per_class, nan=0.0)
    mean_dice = mean_dice_per_class.mean().item()

    # 计算每个类别的IoU
    # 修复：移除不存在的num_classes参数，确保输入格式正确
    iou = compute_iou(
        torch.argmax(all_outputs_softmax, dim=1).unsqueeze(1),  # 添加通道维度
        torch.tensor(all_labels).unsqueeze(1),  # 添加通道维度
        include_background=True
    )
    # 添加nan值处理
    iou = torch.nan_to_num(iou, nan=0.0)
    # 计算每个类别的平均IoU
    # 修复：安全地处理IoU结果，避免索引越界
    mean_iou_per_class = iou.mean(dim=0)
    
    # 修复：确保我们有足够的类别IoU值，不足则用0填充
    if len(mean_iou_per_class) < 3:
        # 创建一个长度为3的张量，用现有值填充，不足部分用0
        temp_iou = torch.zeros(3, device=mean_iou_per_class.device)
        temp_iou[:len(mean_iou_per_class)] = mean_iou_per_class
        mean_iou_per_class = temp_iou
    else:
        mean_iou_per_class = mean_iou_per_class[:3]  # 确保只考虑3个类别
    
    # 新增：打印每个类别的IoU值和mean_iou_per_class的实际形状
    logger.info(f'mean_iou_per_class形状: {mean_iou_per_class.shape}')
    logger.info(f'类别0 IoU: {mean_iou_per_class[0].item():.4f}')
    logger.info(f'类别1 IoU: {mean_iou_per_class[1].item():.4f}')
    logger.info(f'类别2 IoU: {mean_iou_per_class[2].item():.4f}')
    # 计算mIoU并确保结果在合理范围内
    miou = min(mean_iou_per_class.mean().item(), 1.0)  # 确保mIoU不超过1.0

    # 计算整体准确率
    predictions = torch.argmax(all_outputs_softmax, dim=1).numpy()
    accuracy = np.mean(predictions == all_labels)

    # 新增：计算每个类别的Recall和F1
    from sklearn.metrics import recall_score, f1_score
    recall = recall_score(all_labels.flatten(), predictions.flatten(), average=None, labels=[0, 1, 2])
    f1 = f1_score(all_labels.flatten(), predictions.flatten(), average=None, labels=[0, 1, 2])
    mean_recall = recall.mean()
    mean_f1 = f1.mean()

    logger.info(f'类别0 Dice: {mean_dice_per_class[0].item():.4f}')
    logger.info(f'类别1 Dice: {mean_dice_per_class[1].item():.4f}')
    logger.info(f'类别2 Dice: {mean_dice_per_class[2].item():.4f}')
    logger.info(f'平均Dice: {mean_dice:.4f}')

    return total_loss / len(loader.dataset), mean_dice, miou, mean_recall, mean_f1

# 训练模型，增加早停机制
epochs = 300
patience = 20  # 早停 patience
best_val_loss = float('inf')
current_patience = 0
best_dice = 0
best_miou = 0

train_losses = []
val_losses = []
val_dices = []
val_mious = []
val_recalls = []
val_f1s = []

def manage_model_files(model_dir):
    """管理模型文件，只保留4个权重文件：best_model.pth和mIoU值最高的三个文件"""
    # 获取models目录下所有模型文件
    model_files = []
    for file in os.listdir(model_dir):
        if file.endswith('.pth'):
            model_files.append(file)
    
    # 筛选出带mIoU值的模型文件
    miou_model_files = []
    miou_pattern = re.compile(r'best_model_miou_([0-9.]+)\.pth')
    for file in model_files:
        match = miou_pattern.match(file)
        if match:
            miou_value = float(match.group(1))
            miou_model_files.append((file, miou_value))
    
    # 按mIoU值降序排序
    miou_model_files.sort(key=lambda x: x[1], reverse=True)
    
    # 保留mIoU值最高的前三个文件
    keep_files = set(['best_model.pth'])
    for i, (file, _) in enumerate(miou_model_files[:3]):
        keep_files.add(file)
    
    # 删除不需要保留的文件
    for file in model_files:
        if file not in keep_files:
            file_path = os.path.join(model_dir, file)
            try:
                os.remove(file_path)
                logger.info(f'删除模型文件: {file_path}')
            except Exception as e:
                logger.error(f'删除模型文件失败: {file_path}, 错误: {str(e)}')

# 训练模型
logger.info('开始训练模型...')
for epoch in range(epochs):
    logger.info(f'Epoch {epoch+1}/{epochs}')

    # 训练
    train_loss = train_epoch(model, train_loader, optimizer, loss_function, device)
    train_losses.append(train_loss)
    logger.info(f'Train Loss: {train_loss:.4f}')

    # 验证
    val_loss, val_dice, val_miou, val_recall, val_f1 = val_epoch(model, val_loader, loss_function, device)
    val_losses.append(val_loss)
    val_dices.append(val_dice)
    val_mious.append(val_miou)
    val_recalls.append(val_recall)
    val_f1s.append(val_f1)
    logger.info(f'Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val mIoU: {val_miou:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}')

    # 调试信息：检查验证集预测是否全为0
    if val_dice == 0:
        logger.warning('Val Dice为0，可能模型预测全为背景或阈值设置不当')
        # 检查输出分布
        with torch.no_grad():
            sample_input, sample_label = next(iter(val_loader))
            sample_input = sample_input.to(device)
            sample_output = model(sample_input)
            sample_output_sigmoid = torch.sigmoid(sample_output)
            logger.info(f'样本输出最大值: {sample_output_sigmoid.max().item()}')
            logger.info(f'样本输出最小值: {sample_output_sigmoid.min().item()}')
            logger.info(f'样本输出平均值: {sample_output_sigmoid.mean().item()}')
            
            # 尝试不同阈值
            for threshold in [0.1, 0.3, 0.5, 0.7]:
                sample_pred = (sample_output_sigmoid.cpu().numpy() > threshold).astype(np.uint8)
                # 计算这个阈值下的Dice
                intersection = np.sum(sample_pred * sample_label.cpu().numpy())
                union = np.sum(sample_pred) + np.sum(sample_label.cpu().numpy())
                dice = 2 * intersection / (union + 1e-7)
                logger.info(f'阈值 {threshold} 时的样本Dice: {dice.mean():.4f}')
            
            # 可视化第一个样本
            if epoch % 5 == 0:
                sample_image = sample_input[0].cpu().numpy().transpose(1, 2, 0)
                sample_image = (sample_image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
                sample_image = sample_image.astype(np.uint8)
                sample_mask = sample_label[0, 0].cpu().numpy()
                sample_pred = (sample_output_sigmoid[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
                
                # 保存可视化结果
                plt.figure(figsize=(12, 4))
                plt.subplot(131)
                plt.imshow(sample_image)
                plt.title('输入图像')
                plt.axis('off')
                plt.subplot(132)
                plt.imshow(sample_mask, cmap='gray')
                plt.title('真实掩码')
                plt.axis('off')
                plt.subplot(133)
                plt.imshow(sample_pred, cmap='gray')
                plt.title('预测掩码')
                plt.axis('off')
                plt.tight_layout()
                debug_save_path = f'dice_zero_debug_epoch_{epoch}.png'
                plt.savefig(debug_save_path)
                plt.close()
                logger.info(f'已保存Dice为0的调试图像: {debug_save_path}')

    # 更新学习率（基于mIoU）
    scheduler.step(val_miou)

    # 保存最佳模型（基于mIoU）
    if val_miou > best_miou:
        best_miou = val_miou
        # 保存带mIoU值的模型
        model_path = os.path.join(model_dir, f'best_model_miou_{best_miou:.4f}.pth')
        torch.save(model.state_dict(), model_path)
        # 同时保存为best_model.pth
        best_model_path = os.path.join(model_dir, 'best_model.pth')
        torch.save(model.state_dict(), best_model_path)
        logger.info(f'保存最佳模型到: {model_path} 和 {best_model_path}, mIoU: {best_miou:.4f}')
        current_patience = 0  # 重置早停计数器
        # 管理模型文件，只保留4个权重文件
        manage_model_files(model_dir)
    else:
        current_patience += 1
        logger.info(f'早停计数: {current_patience}/{patience}')





    # 早停机制
    if current_patience >= patience:
        logger.info(f'验证性能连续 {patience} 个epoch未提升，提前终止训练')
        break

    # 每10个epoch保存一次模型
    if (epoch + 1) % 10 == 0:
        model_path = os.path.join(model_dir, f'model_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), model_path)
        logger.info(f'保存模型到: {model_path}')

    logger.info('-' * 50)

# 绘制训练曲线
plt.figure(figsize=(18, 12))

plt.subplot(2, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(val_dices, label='Val Dice')
plt.title('Dice Score vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Dice Score')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(val_mious, label='Val mIoU')
plt.title('mIoU vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('mIoU')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(val_recalls, label='Val Recall')
plt.plot(val_f1s, label='Val F1')
plt.title('Recall and F1 vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.legend()

plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()

# 导入预测模块
import predict

# 示例预测
if len(val_images) > 0:
    # 加载最佳模型
    model_path = os.path.join(model_dir, 'best_model.pth')
    test_image_path = val_images[0]
    test_mask_path = val_masks[0]
    
    # 使用predict模块进行预测
    try:
        # 创建结果目录
        if not os.path.exists('test_results_lung7'):
            os.makedirs('test_results_lung7')
        
        # 加载真实掩码
        true_mask = np.array(Image.open(test_mask_path).convert('L'))
        # 新增：定义original_true_mask变量
        original_true_mask = true_mask.copy()
        # 将真实掩码转换为类别
        true_mask = np.zeros_like(true_mask)
        true_mask[original_true_mask == 100] = 1
        true_mask[original_true_mask == 255] = 2
        # 使用predict模块中的函数进行预测
        image, predicted_mask = predict.predict_image(model, test_image_path, val_transform, device)

        # 显示结果
        plt.figure(figsize=(18, 6))
        plt.subplot(141)
        plt.imshow(image)
        plt.title('原始图像')
        plt.axis('off')
        
        plt.subplot(142)
        plt.imshow(true_mask, cmap='viridis')
        plt.title('真实掩码')
        plt.axis('off')
        plt.colorbar(ticks=[0, 1, 2], label='类别')
        
        plt.subplot(143)
        plt.imshow(predicted_mask, cmap='viridis')
        plt.title('预测掩码')
        plt.axis('off')
        plt.colorbar(ticks=[0, 1, 2], label='类别')
        
        # 创建彩色叠加效果
        overlay = image.copy()
        overlay[predicted_mask == 1] = [255, 255, 0]  # 黄色标记轻度病变
        overlay[predicted_mask == 2] = [255, 0, 0]    # 红色标记重度病变
        
        plt.subplot(144)
        plt.imshow(overlay)
        plt.title('预测结果叠加')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('prediction_example.png')
        plt.close()
        logger.info('已保存示例预测结果: prediction_example.png')
    except Exception as e:
        logger.error(f'示例预测失败: {str(e)}')

# 提示用户可以使用predict.py进行测试数据预测
logger.info('模型训练和评估完成!')
logger.info('如需对测试数据进行预测，请运行: python predict.py')

print('Model training and evaluation completed!')
print('To predict on test data, run: python predict.py')