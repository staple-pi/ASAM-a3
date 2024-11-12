import torch
import sys
import cv2
import torch.nn.functional as F
import monai
from torch.utils.data import Dataset, DataLoader
from segment_anything.utils.transforms import ResizeLongestSide
import torch.nn as nn
import torch
from typing import Any, Dict, List, Tuple
#from skimage import transform
from tqdm import tqdm
import numpy as np
import os
from pycocotools import mask as mask_utils
import json
from statistics import mean
join = os.path.join
import torch.nn.init as init
from PIL import Image
from torchvision.transforms import Resize
import torch.distributed as dist

def is_dist_avail_and_initialized():
    """检查是否支持分布式环境"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def is_main_process():
    return get_rank() == 0

def cleanup():
    dist.destroy_process_group()

def reduce_value(value, average=True):
    world_size = get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size

        return value


def get_mask_preprocess_shape(oldh, oldw, long_side_length):
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)

def mask_resieze_pad(mask, target_size, target_long, return_torch=False):
    x, y, = mask.shape
    resized_mask = mask.reshape(1, 1, x, y)
    input_torch = torch.as_tensor(resized_mask, dtype=torch.float)
    output_size = target_size
    downsampled_tensor = torch.nn.functional.interpolate(
        input_torch, size=output_size, mode='bilinear', align_corners=False)
    h, w = downsampled_tensor.shape[-2:]
    padh = target_long - h
    padw = target_long - w
    downsampled_tensor = F.pad(downsampled_tensor, (0, padw, 0, padh))
    if return_torch:
        return downsampled_tensor
    else:
        aarray = downsampled_tensor.numpy()
        return aarray

def mask_preprocess(mask, target_long=1024, return_torch=False):
    target_size = get_mask_preprocess_shape(
        mask.shape[0], mask.shape[1], target_long)
    output = mask_resieze_pad(mask, target_size, target_long, return_torch)
    
    return output[0, :, :, :]


def weights_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        init.xavier_normal_(m.weight)  # 使用 Xavier 初始化方法，也可以根据需要选择其他初始化方法
        if m.bias is not None:
            init.constant_(m.bias, 0)  # 如果模型有偏置项，可以将其初始化为零或其他值

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'  # 通信后端，nvidia GPU推荐使用NCCL
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    dist.barrier()

def box_to_mask(box,mask_shape):
    # 计算box的宽度和高度
    box_width = box[2] - box[0]
    box_height = box[3] - box[1]
    # 计算扩展的大小
    expand_width = int(box_width * 0.15)
    expand_height = int(box_height * 0.15)
    # 更新box的坐标，并确保不超过mask的边界
    new_x1 = int(max(0, box[0] - expand_width))
    new_y1 = int(max(0, box[1] - expand_height))
    new_x2 = int(min(mask_shape[1], box[2] + expand_width))
    new_y2 = int(min(mask_shape[0], box[3] + expand_height))
    # 创建与mask相同形状的数组
    output_array = np.zeros(mask_shape, dtype=int)
    # 填充扩展后的box内部区域的值为1
    output_array[new_y1:new_y2, new_x1:new_x2] = 1
    return output_array

class ASAM(nn.Module):
    def __init__(
        self,
        model,
    ):
        super().__init__()
        self.sam_model = model

    def postprocess_mask(self, low_res_masks):
        masks = F.interpolate(
            low_res_masks,
            (1024, 1024),
            mode="bilinear",
            align_corners=False,
        )
        return masks


    def forward(self, image, bbox, maskin):
        '''
        image: cv2.imread() > cv2.cvtColor() > transform.apply_image() >as tensro>permute>preprocess
        bbox:[x,y,x,y] > transform.apply_boxex > as_tensor > [None,:]
        maskin: mask_preprocess() > as_tensor > [None,:,:,:]
        original_image_size = original_image.shape[:2]  #(高，宽)
        input_size = image before preprocess .shape[-2:]
        '''
        occlude_image_embedding = self.sam_model.image_encoder(image, maskin)
        sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
            points=None,
            boxes=bbox,
            masks=None,
        )
        low_res_masks, iou_predictions = self.sam_model.mask_decoder(
            image_embeddings=occlude_image_embedding,
            image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        upscaled_masks = self.postprocess_mask(low_res_masks)
        return upscaled_masks, occlude_image_embedding


class MaskDiscriminator(nn.Module):
    def __init__(self):
        super(MaskDiscriminator, self).__init__()
        # 输入通道数：3（图像） + 1（掩码） = 4
        input_channels = 4
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),  # [4, 1024, 1024] -> [64, 512, 512]
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),            # [64, 512, 512] -> [128, 256, 256]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),           # [128, 256, 256] -> [256, 128, 128]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),           # [256, 128, 128] -> [512, 64, 64]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),          # [512, 64, 64] -> [1024, 32, 32]
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=4, stride=2, padding=1),         # [1024, 32, 32] -> [2048, 16, 16]
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))                  # [2048, 16, 16] -> [2048, 1, 1]
        self.fc = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )
        
    def forward(self, image, mask):
        # 拼接图像和掩码在通道维度
        x = torch.cat([image, mask], dim=1)  # [batch_size, 4, 1024, 1024]
        x = self.conv1(x)  # [batch_size, 64, 512, 512]
        x = self.conv2(x)  # [batch_size, 128, 256, 256]
        x = self.conv3(x)  # [batch_size, 256, 128, 128]
        x = self.conv4(x)  # [batch_size, 512, 64, 64]
        x = self.conv5(x)  # [batch_size, 1024, 32, 32]
        x = self.conv6(x)  # [batch_size, 2048, 16, 16]
        x = self.global_avg_pool(x)  # [batch_size, 2048, 1, 1]
        x = x.view(x.size(0), -1)    # [batch_size, 2048]
        x = self.fc(x)               # [batch_size, 1]
        
        return x


def train_one_epoch(asam_model, d_model, train_dataloader,epoch,optimizer, optimizer_d, device,batch_size,writer):  ###################

    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True) 
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    bce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    bce_loss2 = nn.BCEWithLogitsLoss(reduction="none")
    mse_loss = torch.nn.MSELoss()
    loss_d = torch.nn.BCELoss()
    mean_loss = torch.zeros(1).to(device)
    mean_dloss = torch.zeros(1).to(device)
    mean_bloss0 = torch.zeros(1).to(device)
    mean_bloss1 = torch.zeros(1).to(device)
    mean_mloss = torch.zeros(1).to(device)
    mean_gloss = torch.zeros(1).to(device)
    mean_cdloss = torch.zeros(1).to(device)
    mean_diloss = torch.zeros(1).to(device)
    asam_model.train()
    d_model.train()
    if is_main_process():
        train_dataloader = tqdm(train_dataloader, file=sys.stdout)
        #input_image[0], bbox_torch, gt_mask, v_mask, maskin_torch, v_64, o_64
    for step, (input_image,input_image_o, bbox_torch, gt_mask, v_mask, maskin_torch, v_64, o_64) in enumerate(train_dataloader):
        optimizer.zero_grad()
        image, image_o,bbox, gt_mask, vmask, maskin, v_64, o_64 = input_image.to(device), input_image_o.to(device),bbox_torch.to(device), gt_mask.to(device), v_mask.to(device), maskin_torch.to(device), v_64.to(device), o_64.to(device)

        asam_pred, image_feature = asam_model(image, bbox, maskin)
        o_rigion = gt_mask - vmask
        o_gt = gt_mask - vmask
        dloss = (seg_loss(asam_pred, gt_mask))# + seg_loss(o_pred, o_gt.float) )
        bloss0 = 10 * bce_loss(asam_pred,gt_mask)
        o_loss = bce_loss2(asam_pred,gt_mask) * o_rigion
        bloss1 = o_loss.sum() / (o_rigion.sum()+1)
        bloss1 = 5*bloss1
        v_feature = image_feature * v_64 #[b,256,64,64]
        o_feature = image_feature * o_64
        input_flat = F.avg_pool2d(o_feature, kernel_size=(64, 64))[:,:,0,0] #[b,256,1,1] > [b,256,0,0]
        target_flat = F.avg_pool2d(v_feature, kernel_size=(64, 64))[:,:,0,0]
        input_flat_new = F.normalize(input_flat, 2, -1)
        target_flat_new = F.normalize(target_flat, 2, -1)
        cd_loss = 0.5 * (1 - (input_flat_new * target_flat_new).sum(-1)).sum()

        g_counts = torch.sum(gt_mask)
        o_counts = torch.sum(o_gt)
        o_rate = o_counts.item() / g_counts.item()
        cd_loss = o_rate * cd_loss
        asam_pred_s = torch.sigmoid(asam_pred)
        g_loss = loss_d(d_model(image_o,asam_pred_s), torch.zeros(size=(batch_size,1),device=device,requires_grad=True))
        loss = dloss + bloss0 + bloss1 + g_loss + cd_loss
        
        loss.backward()
        loss = reduce_value(loss, average=True)
        dloss = reduce_value(dloss, average=True)
        bloss0 = reduce_value(bloss0, average=True)
        bloss1 = reduce_value(bloss1, average=True)
        g_loss = reduce_value(g_loss, average=True)
        cd_loss = reduce_value(cd_loss, average=True)
        
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses
        mean_dloss = (mean_dloss * step + dloss.detach()) / (step + 1)  # update mean losses
        mean_bloss0 = (mean_bloss0 * step + bloss0.detach()) / (step + 1)  # update mean losses
        mean_bloss1 = (mean_bloss1 * step + bloss1.detach()) / (step + 1)  # update mean losses
        mean_gloss = (mean_gloss * step + g_loss.detach()) / (step + 1)  # update mean losses
        mean_cdloss = (mean_cdloss * step + cd_loss.detach()) / (step + 1)  # update mean losses
        
        if is_main_process():
            writer.add_scalar('Loss/sample', loss.item(), epoch * len(train_dataloader) + step)
            writer.add_scalar('dloss/sample', dloss.item(), epoch * len(train_dataloader) + step)
            writer.add_scalar('bloss0/sample', bloss0.item(), epoch * len(train_dataloader) + step)
            writer.add_scalar('bloss1/sample', bloss1.item(), epoch * len(train_dataloader) + step)
            writer.add_scalar('g_loss/sample', g_loss.item(), epoch * len(train_dataloader) + step) 
            writer.add_scalar('cd_loss/sample', cd_loss.item(), epoch * len(train_dataloader) + step) 
        #torch.nn.utils.clip_grad_norm_(asam_model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer_d.zero_grad()
        d_loss = 0.5*loss_d(d_model(image_o,gt_mask), torch.ones(size=(batch_size,1),device=device,requires_grad=True)) + loss_d(d_model(image_o,asam_pred_s.detach()), torch.zeros(size=(batch_size,1),device=device,requires_grad=True))
        d_loss = reduce_value(d_loss, average=True)
        d_loss.backward()
        mean_diloss = (mean_diloss * step + d_loss.detach()) / (step + 1)  # update mean losses
        optimizer_d.step()
        
        if is_main_process():
            train_dataloader.desc = "[epoch {}] mean loss {},dloss {},bloss0 {},bloss1 {},gloss {},cd_loss {},diloss {}".format(epoch, round(mean_loss.item(), 4),
                                    round(mean_dloss.item(), 4),round(mean_bloss0.item(), 4),round(mean_bloss1.item(), 4),
                                    round(mean_gloss.item(), 4),round(mean_cdloss.item(), 4),round(mean_diloss.item(),4))        

    return mean_loss


class SA1BDataset(Dataset):
    def __init__(
        self,
        image_list, image_path, image_path_o,
        trnasform=None,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ):
        self.image_list = image_list
        self.image_path = image_path
        self.image_path_o = image_path_o
        self.transform = trnasform
        self._transform = ResizeLongestSide(1024)
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std

    def __len__(self):
        return len(self.image_list)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.pixel_mean, device=x.device).view(-1, 1, 1)
        std = torch.tensor(self.pixel_std, device=x.device).view(-1, 1, 1)
        x = (x - mean) / std
        # Pad
        h, w = x.shape[-2:]
        padh = 1024 - h
        padw = 1024 - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    

    def __getitem__(self, idx):
        image_filepath = join(self.image_path, self.image_list[idx])
        # image
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_image = self._transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        input_image = self.preprocess(input_image_torch)

        image_filepath_o = join(self.image_path_o, self.image_list[idx])
        image_o = cv2.imread(image_filepath_o)
        image_o = cv2.cvtColor(image_o, cv2.COLOR_BGR2RGB)
        input_image_o = self._transform.apply_image(image_o)
        input_image_o_torch = torch.as_tensor(input_image_o)
        input_image_o_torch = input_image_o_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        input_image_o = self.preprocess(input_image_o_torch)

        annotation_path = image_filepath[:-3] + 'json'
        with open(annotation_path, 'r') as f:
            annotation_data = json.load(f)
        # origin_size
        w, h = annotation_data['image']['width'], annotation_data['image']['height']
        origin_size = (h, w)
        # bbox
        annotation = annotation_data['annotations'][0]
        #bbox_torch = bbox_torch[None, :]   #################
        # gt_mask
        segmentation = mask_utils.decode(annotation['segmentation'])
        gt_mask = mask_preprocess(mask=segmentation, return_torch=True)
        gt_mask = (gt_mask > 0.5).float()

        #gt_mask = gt_mask[None, :, :, :] 
        x, y, w, h = annotation['bbox']
        bbox = np.array([x, y, x + w, y + h])
        maskin = box_to_mask(bbox, segmentation.shape)
        bbox = self._transform.apply_boxes(bbox, origin_size)
        bbox_torch = torch.as_tensor(bbox, dtype=torch.float)
        # maskin
        occluder_mask = mask_utils.decode(annotation['occluder_mask'])
        v_mask = (segmentation & ~occluder_mask)
        omask = segmentation - v_mask
        v_64 = mask_preprocess(mask=v_mask,target_long=64, return_torch=True)
        v_64 = (v_64 > 0.5).float()

        v_mask = mask_preprocess(mask=v_mask, return_torch=True)
        v_mask = (v_mask > 0.5).float()

        occluder_mask = mask_preprocess(mask=occluder_mask, return_torch=True)
        occluder_mask = (occluder_mask > 0.5).float()

        o_64 = mask_preprocess(mask=omask,target_long=64, return_torch=True)
        o_64 = (o_64 > 0.5).float()

        maskin_torch = mask_preprocess(mask = maskin,return_torch=True)
        maskin_torch = (maskin_torch > 0.5).float()
        #maskin = maskin[None, :, :, :]
        
        return input_image[0], input_image_o[0],bbox_torch, gt_mask, v_mask, maskin_torch, v_64, o_64

