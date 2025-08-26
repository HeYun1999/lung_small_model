import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from monai.networks.nets import UNet

# 定义预测函数
def predict_image(model, image_path, transform, device):
    # 加载图像
    image = np.array(Image.open(image_path).convert('RGB'))
    
    # 应用变换
    augmented = transform(image=image)
    image_tensor = augmented['image'].unsqueeze(0).to(device)
    
    # 预测
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        # 取概率最大的类别
        predicted = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    
    # 还原图像（用于显示）
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    image = (image * std + mean) * 255
    image = image.astype(np.uint8)
    
    return image, predicted

# 如果直接运行此文件
if __name__ == '__main__':
    # 配置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = UNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=3,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    
    # 加载模型权重
    model_dir = 'models_lung7'
    model_path = os.path.join(model_dir, 'best_model.pth')
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f'成功加载模型: {model_path}')
    except Exception as e:
        print(f'加载模型失败: {str(e)}')
        exit(1)
    
    # 定义变换
    transform = A.Compose([
        A.Resize(height=512, width=512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    # 检查是否提供了图像路径参数
    import sys
    if len(sys.argv) < 2:
        print('用法: python predict.py <image_path>')
        exit(1)
    
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f'图像文件不存在: {image_path}')
        exit(1)
    
    # 进行预测
    try:
        image, predicted_mask = predict_image(model, image_path, transform, device)
        
        # 显示结果
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.imshow(image)
        plt.title('原始图像')
        plt.axis('off')
        
        plt.subplot(122)
        plt.imshow(predicted_mask, cmap='viridis')
        plt.title('预测掩码')
        plt.axis('off')
        plt.colorbar(ticks=[0, 1, 2], label='类别')
        
        plt.tight_layout()
        # 保存结果
        result_path = 'prediction_result.png'
        plt.savefig(result_path)
        plt.close()
        print(f'预测结果已保存到: {result_path}')
    except Exception as e:
        print(f'预测失败: {str(e)}')