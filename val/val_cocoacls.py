import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import json
import sys
import numpy as np
import torch
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
import cv2
from pycocotools import mask as mask_utils
from collections import defaultdict
from segment_anything.utils.transforms import ResizeLongestSide
from pycocotools.coco import COCO
from PIL import Image, ImageDraw
import random
import numpy as np
from statistics import mean
import torch.nn.init as init
from torch.nn.functional import threshold, normalize
import torch.nn.functional as F 
import argparse
def str2bool(v):
    if isinstance(v,bool):
        return v
    if v.lower() in ('yes','true','t','y','1'):
        return True
    elif v.lower() in ('no','false','f','n','0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')

def read_json(file):
    # 打开 JSON 文件
    with open(file, "r") as f:
        data = json.load(f)
    return data['annotations'][0]

def get_mask_preprocess_shape(oldh, oldw, long_side_length):
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)

def mask_resieze_pad(mask,target_size):
    x,y = mask.shape
    resized_mask = mask.reshape(1,1,x,y)
    input_torch = torch.as_tensor(resized_mask, dtype=torch.float)
    output_size = target_size
    downsampled_tensor = torch.nn.functional.interpolate(input_torch, size=output_size, mode='bilinear', align_corners=False)
    h, w = downsampled_tensor.shape[-2:]
    padh = 1024 - h
    padw = 1024 - w
    downsampled_tensor = F.pad(downsampled_tensor, (0, padw, 0, padh))
    aarray = downsampled_tensor.numpy()
    return aarray

def mask_preprocess(mask):
    target_size = get_mask_preprocess_shape(mask.shape[0],mask.shape[1],1024)
    output = mask_resieze_pad(mask,target_size)
    return output[0,:,:,:]


def calculate_iou(pred, gt):
    # 计算预测值和ground truth之间的交并比（IoU）
    intersection = torch.logical_and(pred, gt).sum().float()
    
    union = torch.logical_or(pred, gt).sum().float()
    iou = intersection / union
    return iou


def polys_to_mask(polygons, height, width):
	rles = mask_utils.frPyObjects(polygons, height, width)
	rle = mask_utils.merge(rles)
	mask = mask_utils.decode(rle)
	return mask

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

def mask_to_bbox(mask):
    rows, cols = np.where(mask == 1)
    if len(rows) == 0 or len(cols) == 0:
        return None
    x_min, x_max = cols.min(), cols.max()
    y_min, y_max = rows.min(), rows.max()
    return [x_min, y_min, x_max, y_max]

def calculate_iou_np(mask_a, mask_b):
    # 计算交集
    intersection = np.logical_and(mask_a, mask_b).sum()
    # 计算并集
    union = np.logical_or(mask_a, mask_b).sum()
    # 计算 IoU
    iou = intersection / union if union != 0 else 0
    return iou

#sam_checkpoint = "E:/code/fine_mask_unet.pth"

def main(args):
    asam_checkpoint = args.asam_checkpoint
    annotations_path = args.annotations_path
    image_dir = args.img_dir
    #sam_checkpoint = None
    model_type = "vit_l"
    device = "cuda"

    ckpt = torch.load(asam_checkpoint,map_location='cuda:0')
    sam = sam_model_registry[model_type](checkpoint= None)
    sam.load_state_dict(ckpt)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    coco = COCO(annotations_path)
    with open(annotations_path,'r') as f:
        data = json.load(f)
    annotations_list =  data['annotations']
    '''
    if args.moiou:
        i=0
        while i < len(annotations_list):
            keys = annotations_list[i].keys()
            if annotations_list[i]['occlude_rate'] ==0:
                del annotations_list[i]
            else:
                i += 1
    '''
    lenth_of_imglist = len(annotations_list)
    num_range = args.num_range
    if num_range > lenth_of_imglist:
        num_range = lenth_of_imglist
    print(num_range)
    batch_no = 0
    total_num = 0
    totalocc_num=0 
    total_ious = 0.
    total_occlu_ious = 0.

    while batch_no < num_range:
        bbox_coords = {}
        occlusion_mask = {}
        ground_truth_masks = {}
        visibel_mask={}
        maskinput = {}
        ious = 0.
        occl_ious = 0.
        num_occ = 0 
        instance_num = 0
        img_size={}
        no_occ={}
        image_path = {}
        print(batch_no)
        next_num = batch_no+50
        if next_num  > lenth_of_imglist:
            next_num = lenth_of_imglist
        for i in range(batch_no, next_num):
            annotation = annotations_list[i]
            image_id = annotations_list[i]['image_id']
            image_info = coco.loadImgs([image_id])[0]
            image_name = image_info['file_name']
            height, width = image_info['height'], image_info['width']
            image_path[i] = os.path.join(image_dir,image_name)
            origin_size = (height,width)
            gt_mask = mask_utils.decode(annotation['segmentation'])
            x1,y1,x2,y2 = mask_to_bbox(gt_mask)
            box = np.array([x1,y1,x2,y2])
            bbox_coords[i] = box
            maskin = box_to_mask(box,origin_size)
            keys = annotation.keys()
            if 'visible_mask' in keys:
                visible_mask = annotation['visible_mask']
                vmask = mask_utils.decode(visible_mask)
            else:
                vmask = gt_mask  #np.zeros_like(gt_mask)
            if args.minus_v:
                maskin = maskin - vmask             ####################################################################################
            if 'invisible_mask' in keys:
                occ_mask = annotation['invisible_mask']
                omask = mask_utils.decode(occ_mask)
            else:
                omask =  np.zeros_like(gt_mask)     
            if annotation['occlude_rate'] < 0.05:
                no_occ[i] = True
                #maskin = np.zeros_like(gt_mask)
                omask = np.zeros_like(gt_mask)    
            else:
                no_occ[i] = False
            visibel_mask[i]  =vmask     
            ground_truth_masks[i] = gt_mask
            occlusion_mask[i] = omask
            maskinput[i] = maskin
            img_size[i] = [height,width]   
        transformed_data = defaultdict(dict)

        # 将图像转换为SAM的格式
        for k in bbox_coords.keys():
            image = cv2.imread(image_path[k])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
            transformed_data[k]['image'] = image
            
        keys = list(bbox_coords.keys())

        for k in keys:
            input_image = transformed_data[k]['image']
            predictor.set_image(input_image)
            prompt_box = bbox_coords[k]
            gt_binary_mask = ground_truth_masks[k]
            #mask_input = np.ones_like(gt_binary_mask)
            mask_input = maskinput[k]
            #mask_input = occlusion_mask[i]
            mask_input = mask_preprocess(mask_input)

            masks_pred, _, _ = predictor.predict(
                box=prompt_box,
                mask_input=mask_input,
                multimask_output=False,
            )
            pred = masks_pred[0]
            gt = gt_binary_mask
            iou = calculate_iou_np(pred, gt)
            ious = ious +iou
            instance_num = instance_num + 1
            total_num = total_num + 1
            if no_occ[k] == False:   #表示没有遮挡
                num_occ+=1
                totalocc_num+=1
                occlusion_pred = (masks_pred[0] == 1) & (visibel_mask[k] == 0)
                occlusion_gt = (gt_binary_mask == 1) & (visibel_mask[k] == 0)
                occlusion_iou = calculate_iou_np(occlusion_pred,occlusion_gt)
                #occl_ious1 = occl_ious1 + occlusion_iou1
                occl_ious = occl_ious + occlusion_iou

        mIoU = (ious / instance_num)#.float()
        occlu_mIoU = (occl_ious / num_occ)#.float()
        print('miou'+ str(mIoU))
        print('occlu_miou'+ str(occlu_mIoU))

        total_ious = total_ious + ious
        total_occlu_ious = total_occlu_ious + occl_ious
        batch_no = batch_no + 50

    mIoU = (total_ious / total_num)#.float()
    occlu_mIoU = (total_occlu_ious / totalocc_num)#.float()
    print('miou'+ str(mIoU))
    print('occlu_miou'+ str(occlu_mIoU))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--asam_checkpoint', type=str, default= "E:/code/asam-2w-0.pth")   # train_o weight 的地址
    parser.add_argument('--img_dir',type=str,default='E:/code/COCOA/val2014')      # COCOA-val的地址
    parser.add_argument('--annotations_path',type=str,default= 'E:/code/COCOA/annotations/COCO_amodal_val2014_with_classes.json')
    parser.add_argument('--num_range',type=int,default=5000)
    parser.add_argument('--minus_v',type=str2bool,default=True)
    opt = parser.parse_args()
    print("minus_v:", opt.minus_v)
    main(opt)