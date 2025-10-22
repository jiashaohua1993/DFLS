import torch
import torch.nn.functional as F
from torchvision import transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def get_edges(image):
    # 使用Sobel算子提取边缘
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0).to(image.device)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).float().unsqueeze(0).unsqueeze(0).to(image.device)

    edges_x = F.conv2d(image, sobel_x, padding=1)
    edges_y = F.conv2d(image, sobel_y, padding=1)

    # 计算梯度幅度
    edges = torch.sqrt(edges_x ** 2 + edges_y ** 2)
    return edges


def edge_loss(prediction, target):
    # 将图像转换为torch张量并调整维度
    # prediction = transforms.ToTensor()(prediction).unsqueeze(0)
    # target = transforms.ToTensor()(target).unsqueeze(0)

    # 将图像转换为灰度
    prediction = transforms.Grayscale()(prediction)
    target = transforms.Grayscale()(target)

    # 获取边缘
    edges_pred = get_edges(prediction)
    edges_target = get_edges(target)

    # 计算边缘损失（均方误差）
    loss = F.mse_loss(edges_pred, edges_target)
    return loss
