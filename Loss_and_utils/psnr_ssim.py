import torch
import torch.nn.functional as F


def calculate_psnr(img1, img2):
    """
    计算PSNR（峰值信噪比）
    数据范围 [0, 1]
    """
    img1 = img1.clamp(0, 1)
    img2 = img2.clamp(0, 1)

    mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])

    # 防止除以零
    mse = torch.where(mse == 0, torch.tensor(1e-10, device=mse.device), mse)

    PIXEL_MAX = 1.0
    return 20 * torch.mean(torch.log10(PIXEL_MAX / torch.sqrt(mse)))
