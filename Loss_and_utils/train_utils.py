import torch
import os
import torchvision.utils as vutils


def save_models( model_en,model_de,epoch,index,ckpts_path):
    torch.save(model_en.state_dict(), os.path.join(ckpts_path, f'FVSRs_en_epo_{epoch}_ind{index}.pth'))
    torch.save(model_de.state_dict(), os.path.join(ckpts_path, f'FVSRs_de_epo_{epoch}_ind{index}.pth'))
   # print(f'Epoch {epoch} completed. Models saved.')

def save_de_out_image(de_out, video_name, index, epoch,sample_path):
    # 保存每个图像

    for i in range(de_out.size(0)):  # 遍历 batch 中的每个图像
        image_name = f'epo{epoch}_bat{index}_img{i}_{video_name[0][:8]}.png'
        save_path = os.path.join(sample_path, image_name)
        vutils.save_image(de_out[i], save_path, normalize=True)
      #  print(f'Saved image: {sample_path}')

    # 将 epoch, index, video_name 写入到 txt 文件中
    # 日志文件路径
    log_file_path = os.path.join(sample_path, 'log.txt')

    # 仅在文件不存在时创建
    if not os.path.exists(log_file_path):
        with open(log_file_path, 'w') as log_file:  # 创建新文件
            log_file.write('Epoch,  Index,  Video Name\n')  # 可选：写入表头

    # 写入日志信息
    with open(log_file_path, 'a') as log_file:  # 以追加模式打开文件
        log_file.write(f'Epoch: {epoch}, Index: {index}, Video Name: {video_name}\n')



def save_input_image(input, index, epoch,sample_path):
    # 保存每个图像
    for i in range(input.size(0)):  # 遍历 batch 中的每个图像
        image_name = f'epo{epoch}_bat{index}_img{i}.png'
        save_path = os.path.join(sample_path, image_name)
        vutils.save_image(input[i], save_path, normalize=True)
       # print(f'Saved image: {sample_path}')
def save_gt_image(input, index, epoch,sample_path):
    # 保存每个图像
    for i in range(input.size(0)):  # 遍历 batch 中的每个图像
        image_name = f'epo{epoch}_bat{index}_img{i}_gt.png'
        save_path = os.path.join(sample_path, image_name)
        vutils.save_image(input[i], save_path, normalize=True)
        #print(f'Saved image: {sample_path}')

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            pass
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)




