import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def get_edges(image):
    # 使用Sobel算子提取边缘
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).float().unsqueeze(0).unsqueeze(0)

    edges_x = F.conv2d(image, sobel_x, padding=1)
    edges_y = F.conv2d(image, sobel_y, padding=1)

    # 计算梯度幅度
    edges = torch.sqrt(edges_x ** 2 + edges_y ** 2)
    return edges


def edge_loss(prediction, target):
    # 将图像转换为torch张量并调整维度
    prediction = transforms.ToTensor()(prediction).unsqueeze(0)
    target = transforms.ToTensor()(target).unsqueeze(0)

    # 将图像转换为灰度
    prediction = transforms.Grayscale()(prediction)
    target = transforms.Grayscale()(target)

    # 获取边缘
    edges_pred = get_edges(prediction)
    edges_target = get_edges(target)

    # 计算边缘损失（均方误差）
    loss = F.mse_loss(edges_pred, edges_target)

    # 返回边缘图像和损失
    return edges_pred, edges_target, loss


# 示例使用
# 加载图像
predicted_image = Image.open('/home/user/projects/codes/HK/lab-to-student/lab13_New_VDTR/FVSRs_V5/ckpts/samples/epo0_bat6900_img0.png').convert('RGB')
target_image = Image.open('/home/user/projects/codes/HK/lab-to-student/lab13_New_VDTR/FVSRs_V5/ckpts/samples/epo0_bat6900_img0_gt.png').convert('RGB')

# 计算边缘损失
edges_pred, edges_target, loss = edge_loss(predicted_image, target_image)
print(f'Edge Loss: {loss.item()}')

# 转换边缘图像为NumPy数组并展示
edges_pred_np = edges_pred.squeeze().detach().numpy()
edges_target_np = edges_target.squeeze().detach().numpy()

# 使用plt展示边缘图像
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(edges_pred_np, cmap='gray')
plt.title('Predicted Edges')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(edges_target_np, cmap='gray')
plt.title('Target Edges')
plt.axis('off')

plt.show()
