import argparse
import os
import torch
from Data  import data_preprocess
from Network.FVSRs import FVSRs_en,FVSRs_de

import time
import torchvision.utils as vutils

def run_inference(model_ckpt, output_dir):
    # 加载模型
    model_en = FVSRs_en().cuda()  # 使用 .cuda() 替代 .to(device)
    model_de = FVSRs_de().cuda()  # 使用 .cuda() 替代 .to(device)
    if torch.cuda.is_available():
        device = f"cuda:0"
    else:
        device = torch.device("cpu")
    #if model_ckpt != 'None':
    ckpt = torch.load(model_ckpt,map_location=device)
    model_en.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['enc'].items()})
    model_de.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['dec'].items()})
    del ckpt
    # model_en.load_state_dict(torch.load(model_en_path, map_location='cuda:0'))
    model_en.eval()
    model_de.eval()

    for index, batch_data in enumerate(dataloder):
        input_fr = batch_data["input_frames"].cuda()  # 使用 .cuda()
        start_time=time.time()
        with torch.no_grad():
            en_output = model_en(input_fr)
            de_output ,diff = model_de(en_output, input_fr)
        end_time=time.time()
        reference_time=end_time-start_time
        print(reference_time)

        save_path_base = os.path.abspath(os.path.join(output_dir, batch_data["video_name"][0]))
        os.makedirs(save_path_base, exist_ok=True)
        save_path = os.path.join(save_path_base, batch_data["save_name"][0])
        vutils.save_image(de_output, save_path, normalize=True)  # antialias=True,

def video_to_frame():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on VSR model")
    parser.add_argument("--data_path", type=str, default=r'/home/user/projects/codes/HK/lab-to-student/lab18_SPL/dsp eval/te data', help="Path to the decoder model file")
    parser.add_argument("--model_ckpt", type=str,default=r'ckpts/samples261_1000.pth')
    parser.add_argument("--output_dir", type=str, default=r'Results')
    # parser.add_argument('--gt_folder', type=str, default='sharp')
    parser.add_argument('--latent', type=str, default='blur_64', help='the input folder')
   # parser.add_argument('--rb', type=str, default='rb', help='the input folder')
    parser.add_argument("--is_train", default=False)
    parser.add_argument("--nodes", type=int, default=1, help="number of total node")
    parser.add_argument("--batch_size", type=int, default=1, help="number of batch size")
    parser.add_argument("--works", type=int, default=0, help="number of works")
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--interval", type=int, default=1)

    parser.add_argument('--label_smoothing', type=float, default=0.001, help='Label smoothing factor')
    parser.add_argument("--fusion_in_channel", type=int, default=48)
    parser.add_argument("--patch", type=int, default=4)
    parser.add_argument("--num_blocks", type=int, default=12)

    parser.add_argument("--rate", type=int, default=4)
    #parser.add_argument("--frame_size", type=int, default=12)

    args = parser.parse_args()

    current_path = os.path.abspath(__file__)
    # 获取当前文件所在的目录
    directory = os.path.dirname(current_path)

    args.model_ckpt = os.path.join(directory, args.model_ckpt)
    args.output_dir = os.path.join(directory, args.output_dir)

    dataset = data_preprocess.data(args)
    dataloder = torch.utils.data.DataLoader(dataset)
    run_inference(args.model_ckpt, args.output_dir)



