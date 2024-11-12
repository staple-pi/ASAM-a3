from typing import List, Optional, Tuple
import sys
import torch
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor, sam_model_registry_o
from statistics import mean
import torch.nn.init as init
from torch.nn.functional import threshold, normalize
import os
import tempfile
import argparse
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch
#from skimage import transform
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
from statistics import mean
join = os.path.join
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from asam_utils_o import init_distributed_mode, weights_init, ASAM, SA1BDataset, cleanup, MaskDiscriminator,train_one_epoch

def str2bool(v):
    if isinstance(v,bool):
        return v
    if v.lower() in ('yes','true','t','y','1'):
        return True
    elif v.lower() in ('no','false','f','n','0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')
    
def main(args):
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")
    #read args
    # 初始化各进程环境
    init_distributed_mode(args=args)

    rank = args.rank
    if rank == 0 :
        print(args)
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        tb_writer = SummaryWriter()
    else:
        tb_writer = None

    device = torch.device(args.device)
    #args.lr *= args.world_size  # 学习率要根据并行GPU的数量进行倍增
    #set model
    model_type = "vit_l"
    asam_model = sam_model_registry[model_type](checkpoint=None)
    
    pretrained_state_dict = torch.load(args.sam_checkpoint)
    train_layers=[]

    if args.pretrain_use:
        model_weight = args.asam_checkpoint
        new_weight_dict = torch.load(model_weight, map_location=device)
        asam_model.load_state_dict(new_weight_dict, strict=False)
        for k in new_weight_dict.keys():
            if k in pretrained_state_dict.keys()and not k.startswith('mask_decoder.output_hypernetworks_mlps'):
                continue
                #new_weight_dict[k] = pretrained_state_dict['model'][k]
            else:
                train_layers.append(k) 
    else:   #如果没有使用预训练权重，所有进程同步初始化的权重
        model_weight = None
        checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
        asam_model.apply(weights_init)
        new_weight_dict = asam_model.state_dict()
        for k in new_weight_dict.keys():
            if k in pretrained_state_dict.keys()and not k.startswith('mask_decoder.output_hypernetworks_mlps'):
                new_weight_dict[k] = pretrained_state_dict[k]
            else:
                train_layers.append(k)         
        asam_model.load_state_dict(new_weight_dict, strict=False)
        if rank ==0:
            torch.save(asam_model.state_dict(),checkpoint_path)
        dist.barrier()
        asam_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    #seg training parameters  
    params=[]
    for name, param in asam_model.named_parameters():
        if any(name.startswith(prefix) for prefix in train_layers):
            if rank == 0:
                print(name)
            params.append(param)
            param.requires_grad = True
        else:
            param.requires_grad = False   

    asam = ASAM(model=asam_model).to(device=device)
    asam.train()
    d_model = MaskDiscriminator().to(device=device)
    d_model_weight=torch.load(args.discriminator_checkpoint, map_location=device)
    d_model.load_state_dict(d_model_weight,strict=False)
    d_model.train()
    asam = torch.nn.parallel.DistributedDataParallel(asam, device_ids=[args.gpu],find_unused_parameters=True)
    d_model = torch.nn.parallel.DistributedDataParallel(d_model, device_ids=[args.gpu],broadcast_buffers=False)
    #set dataset
    img_list = os.listdir(args.data_dir)    
    img_list = [img for img in img_list if img.endswith(".jpg")]
    img_list = img_list[:args.data_num]
    train_dataset = SA1BDataset(img_list, args.data_dir, args.data_dir_o)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
    if rank == 0:
        print('Using {} dataloader workers every process'.format(nw))
        print("Number of training samples: ", len(train_dataset))
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler = train_batch_sampler,
        pin_memory=True,
        num_workers=nw,
        )
    optimizer = torch.optim.AdamW(params,lr=args.lr,weight_decay=0.001)
    optimizer_d = torch.optim.Adam(d_model.parameters(),lr=2*args.lr,weight_decay=0.001)
    scheduler = CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=args.end_lr)
    scheduler2 = CosineAnnealingLR(optimizer_d,T_max=args.epochs,eta_min=args.end_lr)
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        mean_loss = train_one_epoch(asam, d_model,train_dataloader, epoch, optimizer, optimizer_d, device, args.batch_size, tb_writer)
        scheduler.step()
        scheduler2.step()
        if rank == 0:
            tags = ["loss","learning_rate"]
            tb_writer.add_scalar(tags[0], mean_loss, epoch)
            tb_writer.add_scalar(tags[1], optimizer.param_groups[0]["lr"], epoch)
            #torch.save(model.module.state_dict(), "./weights/model-{}.pth".format(epoch))
            num_epoch = int(epoch / 2)
            weight_name ="asam-a3-{}.pth".format(num_epoch)
            d_weight_name ="discriminator-a3-{}.pth".format(num_epoch)
            torch.save(asam.module.sam_model.state_dict(), os.path.join(args.weight_savepath, weight_name))
            torch.save(d_model.module.state_dict(), os.path.join(args.weight_savepath, d_weight_name))
    if rank == 0:
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)

    cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sam_checkpoint', type=str, default= "/data/checkpoint/sam_vit_l_0b3195.pth")   # 加载初始SAM权重 命名为：sam_vit_l_0b3195.pth
    parser.add_argument('--asam_checkpoint', type=str, default= '/data/checkpoint/asam-0.pth')                          # 不使用预训练权重
    parser.add_argument('--discriminator_checkpoint', type=str, default="/data/checkpoint/sa1b_discriminator1.pth")     # 使用初始的discriminator权重，命名为:sa1b_discriminator1.pth
    parser.add_argument('--weight_savepath', type=str, default= "/data/checkpoint")
    parser.add_argument('--data_dir',type=str,default="/data1/zb/SA1B-a")      #shiyon
    parser.add_argument('--data_dir_o',type=str,default='/data1/zb/SA1B-o')
    parser.add_argument('--data_num',type=int,default = 40000)          #使用4w数据量
    parser.add_argument('--epochs', type=int, default = 15)
    parser.add_argument('--batch-size', type=int, default = 1)
    parser.add_argument('--lr', type=float, default = 8e-4)
    parser.add_argument('--end_lr', type=float, default = 5e-5)
    # 是否启用SyncBatchNorm
    parser.add_argument('--syncBN', type=str2bool, default=False)
    parser.add_argument('--pretrain_use', type=str2bool, default=False)   
    parser.add_argument('--freeze-layers', type=str2bool, default=False)
    # 不要改该参数，系统会自动分配
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    opt = parser.parse_args()

    main(opt)