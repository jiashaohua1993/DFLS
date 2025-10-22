import argparse
import torch
from torch.nn import MSELoss
from Data import data_preprocess
from Network.FVSRs import FVSRs_de,FVSRs_en
from Loss_and_utils import train_utils,loss
from torch.optim.lr_scheduler import StepLR
from einops import rearrange
import numpy as np
from Network import e_augment
from Loss_and_utils import edge_loss
import skimage.metrics
from copy import deepcopy
policy = "color,translation,cutout"
policy2="color"

# 在 Trainer 类中，添加 L2 正则化损失
def l2_regularization(model, lambda_l2):
    l2_loss = 0
    for param in model.parameters():
        l2_loss += torch.norm(param) ** 2
    return lambda_l2 * l2_loss


def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 数据集和数据加载器
        self.dataset = data_preprocess.data(args)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=args.batch_size, shuffle=True,num_workers=args.works)

        # 初始化模型
        self.model_en = FVSRs_en().to(self.device).apply(train_utils.weights_init)
        self.model_de = FVSRs_de().to(self.device).apply(train_utils.weights_init)
        self.Discriminator=loss.Discriminator().to(self.device).apply(train_utils.weights_init)



        # 损失函数和优化器

        self.Gan_loss=torch.nn.BCEWithLogitsLoss().to(self.device)
        self.mse_loss=torch.nn.MSELoss().to(self.device)
        self.diff_loss = torch.nn.MSELoss().to(self.device)
        self.optimizer_en = torch.optim.Adam(self.model_en.parameters(), lr=0.00001, betas=(0.9, 0.999))
        self.optimizer_de = torch.optim.Adam(self.model_de.parameters(), lr=0.00001, betas=(0.9, 0.999))
        self.optimizer_Dis = torch.optim.Adam(self.Discriminator.parameters(), lr=0.00001, betas=(0.9, 0.999))

        # 学习率调度器
        self.scheduler_en = StepLR(self.optimizer_en, step_size=10, gamma=0.98)
        self.scheduler_de = StepLR(self.optimizer_de, step_size=10, gamma=0.98)
        self.scheduler_Dis = StepLR(self.optimizer_Dis, step_size=10, gamma=0.98)
        checkpoint = args.ckpt
        gancheckpoint = args.ganckpt
        if checkpoint != 'None':
            ckpt = torch.load(checkpoint)
            ganckpt = torch.load(gancheckpoint)
            self.model_en.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['enc'].items()})
            self.model_de.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['dec'].items()})
            self.Discriminator.load_state_dict({k.replace('module.', ''): v for k, v in ganckpt['ganc'].items()})
            current_iteration = int(checkpoint.split('_')[-1].split('.')[0])
            del ckpt

        # checkpoint = args.ckpt
        # gancheckpoint=args.ganckpt
        # if checkpoint != 'None':
        #     ckpt = torch.load(checkpoint)
        #     ganckpt = torch.load(gancheckpoint)
        #     self.model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['codec'].items()})
        #     self.Discriminator.load_state_dict({k.replace('module.', ''): v for k, v in ganckpt['ganc'].items()})
        #     current_iteration = int(checkpoint.split('_')[-1].split('.')[0])
        #     del ckpt

        self.ema_encoder = copy_G_params(self.model_en)
        self.ema_decoder = copy_G_params(self.model_de)
        self.ema_discriminator = copy_G_params(self.Discriminator)

    def train(self):
        for epo in range(self.args.epoch):
            num_batches = len(self.dataloader)
            for index, batch_data in enumerate(self.dataloader):
                input_fr = batch_data["input_frames"].to(self.device)

                # input_fr_4 = rearrange(input_fr, 'b t c n d -> (b t) c n d')[::5]
                gt = batch_data["gt_frames"].to(self.device)
                gt = rearrange(gt, 'b t c n d -> (b t) c n d')
                #gt_a=e_augment.DiffAugment(gt, policy2)

                video_name=batch_data['video_name']
                # 前向传播与反向传播
                self.model_en.zero_grad()
                en_output = self.model_en(input_fr)

                self.model_de.zero_grad()
                de_out,diff = self.model_de(en_output,input_fr)

                self.Discriminator.zero_grad()

                # 真实样本
                target_score = self.Discriminator(gt)
                dis_real = self.Gan_loss(target_score, torch.ones_like(target_score))
                dis_real.backward()

                chance = np.random.rand(1)
                if chance < 0.5:
                    current_score = self.Discriminator(de_out.detach())
                else:
                    fake_img = e_augment.DiffAugment(gt.clone(), policy)
                    current_score = self.Discriminator(fake_img)


                dis_fake = self.Gan_loss(current_score, torch.zeros_like(current_score))
                dis_fake.backward()
                self.optimizer_Dis.step()


                update_score = self.Discriminator(de_out)
                generate_loss = self.Gan_loss(update_score, torch.ones_like(update_score)) * 0.0008
                mse=self.mse_loss(de_out,gt)
                (mse+generate_loss).backward()

                self.optimizer_en.step()
                self.optimizer_de.step()

                for p, avg_p in zip(self.model_en.parameters(), self.ema_encoder):
                    avg_p.mul_(0.999).add_(0.001 * p.data)
                for p, avg_p in zip(self.model_de.parameters(), self.ema_decoder):
                    avg_p.mul_(0.999).add_(0.001 * p.data)
                for p, avg_p in zip(self.Discriminator.parameters(), self.ema_discriminator):
                    avg_p.mul_(0.999).add_(0.001 * p.data)

                if index % 300 == 0:
                    load_params(self.model_en, self.ema_encoder)
                    load_params(self.model_de, self.ema_decoder)
                    load_params(self.Discriminator, self.ema_discriminator)

                if index % 1000==0:
                    #print(f'Epoch {epo}, Batch/num_batches {index}/{num_batches},diff_loss: {diff_loss.item():.7f},generate_loss: {generate_loss.item():.7f}')
                    print(f'Epoch {epo}, Batch/num_batches {index}/{num_batches},MSELoss: {mse.item():.7f},generate_loss: {generate_loss.item():.7f}')

                # 输出损失
                if index % 1000 == 0:
                    train_utils.save_input_image(de_out, index, epo, args.sample_path)
                    train_utils.save_de_out_image(input_fr.add(1).mul(0.5), video_name, index, epo, args.sample_path)
                    train_utils.save_gt_image(gt, index, epo, args.sample_path)
                    # print(f'Epoch {epo}, Batch {index}, MSELoss: {mse.item():.7f},g0nerate_loss: {generate_loss.item():.7f}')

            # 保存模型
                #if index % 3000 ==0:
                if index % 3000 == 0:
                    torch.save({'enc': self.model_en.state_dict(), 'dec': self.model_de.state_dict()},args.ckpts_path + '%d_%d.pth' % (epo, index))
                    torch.save({'ganc': self.Discriminator.state_dict()}, 'Gan' + '%d_%d.pth' % (epo, index))
                               #'/%d.pth' % index)

                    #train_utils.save_models(self.model_en,self.model_de, epo, index, args.ckpts_path)
                #if index !=0 and index % 20000 ==0:
            self.scheduler_en.step()
            self.scheduler_de.step()
            self.scheduler_Dis.step()


parser = argparse.ArgumentParser(description="Train FVSRs model")
parser.add_argument("--data_path", default=r"/home/user/projects/codes/HK/data/face_test/Face_1024_frames")
parser.add_argument("--ckpts_path", default=r"/home/user/projects/codes/HK/lab-to-student/lab18_SPL/VSR05/ckpts/samples")
parser.add_argument("--sample_path", default=r"/home/user/projects/codes/HK/lab-to-student/lab18_SPL/VSR05/ckpts/samples")
parser.add_argument("--is_train", default=True)
parser.add_argument("--nodes", type=int, default=1, help="number of total node")
parser.add_argument("--batch_size", type=int, default=5, help="number of batch size")
parser.add_argument("--works", type=int, default=0, help="number of works")
parser.add_argument("--num_frames", type=int, default=5)
parser.add_argument("--interval", type=int, default=1)
parser.add_argument("--epoch", type=int, default=2001)
parser.add_argument('--gt_folder', type=str, default='sharp', help='the ground truth folder')
parser.add_argument('--latent', type=str, default='sharp', help='the input folder')
# parser.add_argument('--rb', type=str, default='rb_4', help='the input folder')
parser.add_argument('--label_smoothing', type=float, default=0.001, help='Label smoothing factor')
parser.add_argument("--fusion_in_channel", type=int, default=48)
parser.add_argument("--patch", type=int, default=4)
parser.add_argument("--num_blocks", type=int, default=12)
parser.add_argument("--ckpt", type=str, default='/home/user/projects/codes/HK/lab-to-student/lab18_SPL/VSR05/ckpts/samples66_0.pth')
parser.add_argument("--ganckpt", type=str, default='/home/user/projects/codes/HK/lab-to-student/lab18_SPL/VSR05/ckpts/Gan66_0.pth')
# parser.add_argument("--ckpt", type=str, default='None')

args = parser.parse_args()


if __name__ == "__main__":

    # 实例化 Trainer 并开始训练
    trainer = Trainer(args)
    trainer.train()