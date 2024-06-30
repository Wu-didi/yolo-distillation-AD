#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import itertools
import os
import random
from tqdm import tqdm
from EfficientAD.common import get_autoencoder, get_pdn_small, get_pdn_medium, \
    ImageFolderWithoutTarget, ImageFolderWithPath, InfiniteDataloader, student_model
from sklearn.metrics import roc_auc_score


from models.yolo import Model
from utils.general import check_yaml
from utils.torch_utils import select_device
from utils.datasets import LoadImages
from models.experimental import attempt_load
import torch.nn.functional as F


# 解析YOLO标注文件
def parse_yolo_annotation(annotation_file):
    with open(annotation_file, 'r') as file:
        lines = file.readlines()
    annotations = []
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        x, y, width, height = map(float, parts[1:])
        annotations.append((class_id, x, y, width, height))
    return annotations


# 过滤标签中的文件,返回true or false，如果是true,则忽略这个标签
def is_ignore(x1,x2,y1,y2, threshold = 16):
    # 判断一下，如果x2-x1 或者 y2-y1 小于16, 则不进行裁剪,跳过这个边界框
    if x2 - x1 < threshold or y2 - y1 < threshold:
        return True
    # 保证都大于0，如果有一个小于的就跳过
    if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
        return True
    return False

def xywh2xyxy(x, y, w, h, img_w, img_h):
    x1 = int((x - w / 2) * img_w)
    y1 = int((y - h / 2) * img_h)
    x2 = int((x + w / 2) * img_w)
    y2 = int((y + h / 2) * img_h)
    return x1, y1, x2, y2
    
# 得到缩放后坐标
def get_scaled_coordinates(x1, y1, x2, y2, feature, img):
    # 计算一下缩放尺度
    scale_x = feature.shape[3] / img.shape[3]
    scale_y = feature.shape[2] / img.shape[2]
    
    if scale_x != scale_y:
        print(imgsz)
        raise ValueError("Scaling ratios are not equal.")
    
    # Rescale xyxy coordinates to match feature size
    x1_scaled = int(x1 * scale_x)
    y1_scaled = int(y1 * scale_y)
    x2_scaled = int(x2 * scale_x)
    y2_scaled = int(y2 * scale_y)
    return x1_scaled, y1_scaled, x2_scaled, y2_scaled
    
# 获取crop后的feature和image
def get_crop(x1, y1, x2, y2, feature, img):
    # Rescale xyxy coordinates to match feature size
    x1_scaled, y1_scaled, x2_scaled, y2_scaled = get_scaled_coordinates(x1, y1, x2, y2, feature, img)
    # Crop feature using xyxy coordinates
    cropped_feature = feature[:, :, int(y1_scaled):int(y2_scaled), int(x1_scaled):int(x2_scaled)]
    feature1_resized = F.interpolate(cropped_feature, size=(65, 65), mode='bilinear', align_corners=False)

    # 裁剪图像
    cropped_img = img[:, :, int(y1):int(y2), int(x1):int(x2)]
    # print("cropped_img shape: ", cropped_img.shape)  # 裁剪的图像shape是变化的，需要resize到256*256, 但是特征图是56*56
    cropped_img = F.interpolate(cropped_img, size=(256, 256), mode='bilinear', align_corners=False)
    return feature1_resized, cropped_img



def my_test(test_set, teacher, student, teacher_mean, teacher_std,
         q_st_start, q_st_end,
         desc='Running inference',test_output_dir=None, device="cuda"):
    y_true = []
    y_score = []
    for path, img, im0s, vid_cap in tqdm(test_set, desc=desc):
        _, img_h, img_w = img.shape
        # process image
        img = torch.from_numpy(img).to(device)
        half = False
        img= img.half() if half else  img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        
        # teacher_yolo inference
        with torch.no_grad():
            pred = teacher(img, augment=False, visualize=False)
            extract_feature = pred[-1]
            feature = extract_feature[0]
        
        # 获取names
        names = teacher.module.names if hasattr(teacher, 'module') else teacher.names  # get class names
        
        
        # 找到对应的label文件
        txt_path = str(path).replace('images', 'labels').replace('.jpg', '.txt')
        # 解析标签文件，获得标签信息
        annotations = parse_yolo_annotation(txt_path)
        for annotation in annotations:
            class_id, x, y, width, height = annotation
            
            # 针对class_id进行判断，如果id>10, 则跳过
            if int(class_id) not in [8,9]:
                continue
            
            # 计算边界框的坐标
            x1, y1, x2, y2 = xywh2xyxy(x, y, width, height, img_w, img_h)
            # 过滤掉一些不符合要求的边界框
            if is_ignore(x1, x2, y1, y2, threshold=h_w_threshold):
                continue
            
            # 获取crop后的feature和image
            cropped_feature, cropped_img = get_crop(x1, y1, x2, y2, feature, img)
            # 对裁剪的feature 进行正则化
            teacher_output_st = (cropped_feature - teacher_mean[:,:teacher_out_channels,:,:]) / teacher_std[:,:teacher_out_channels,:,:]
                
            # 开始训练学生模型
            student_output_st = student(cropped_img)[:, :teacher_out_channels]  #  torch.Size([1, 768, 56, 56])
        
            # 计算差异
            map_combined, map_st = my_predict(
                teacher_output_st=teacher_output_st, student_output_st=student_output_st,
                )
            y_true_image = 0 if 'good' in str(names[class_id])  else 1
            # print(map_st.shape)
            map_st_cpu = map_st.cpu().numpy()
            y_score_image = np.max(map_st_cpu)
            y_true.append(y_true_image)
            y_score.append(y_score_image)
    auc = roc_auc_score(y_true=y_true, y_score=y_score)
    print("auc:", auc * 100)
    return auc * 100


@torch.no_grad()
def my_predict(teacher_output_st, student_output_st):
    map_st = torch.mean((teacher_output_st - student_output_st)**2,
                        dim=1, keepdim=True)
    map_combined = map_st
    return map_combined, map_st



def my_map_normalization(validation_loader, teacher, student,
                      teacher_mean, teacher_std, desc='Map normalization',device="cuda"):
    
    maps_st = []
    for path, img, im0s, vid_cap in tqdm(validation_loader, desc=desc):
        _, img_h, img_w = img.shape
        # process image
        img = torch.from_numpy(img).to(device)
        half = False
        img= img.half() if half else  img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        
        # teacher_yolo inference
        with torch.no_grad():
            pred = teacher(img, augment=False, visualize=False)
            extract_feature = pred[-1]
            feature = extract_feature[0]
        
        # 找到对应的label文件
        txt_path = str(path).replace('images', 'labels').replace('.jpg', '.txt')
        # 解析标签文件，获得标签信息
        annotations = parse_yolo_annotation(txt_path)
        for annotation in annotations:
            class_id, x, y, width, height = annotation
                        
            # 针对class_id进行判断，如果id>9, 则跳过
            if int(class_id) not in [8,9]:
                continue
            # 计算边界框的坐标
            x1, y1, x2, y2 = xywh2xyxy(x, y, width, height, img_w, img_h)
            # 过滤掉一些不符合要求的边界框
            if is_ignore(x1, x2, y1, y2, threshold=h_w_threshold):
                continue
            
            # 获取crop后的feature和image
            cropped_feature, cropped_img = get_crop(x1, y1, x2, y2, feature, img)
            
            # 开始训练学生模型
            student_output_st = student(cropped_img)[:, :teacher_out_channels]  #  torch.Size([1, 768, 56, 56])
            # 对裁剪的feature 进行正则化
            teacher_output_st = (cropped_feature - teacher_mean[:,:teacher_out_channels,:,:]) / teacher_std[:,:teacher_out_channels,:,:]
                
            # 计算差异
            map_combined, map_st = my_predict(
                teacher_output_st=teacher_output_st, student_output_st=student_output_st,
                )
            maps_st.append(map_st)
    maps_st = torch.cat(maps_st)
    q_st_start = torch.quantile(maps_st, q=0.9)
    q_st_end = torch.quantile(maps_st, q=0.995)
    return q_st_start, q_st_end

@torch.no_grad()
def teacher_normalization(teacher, train_loader):

    mean_outputs = []
    for train_image, _ in tqdm(train_loader, desc='Computing mean of features'):
        if on_gpu:
            train_image = train_image.cuda()
        # print("train_image: ", train_image.shape)  # torch.Size([1, 3, 256, 256]
        teacher_output = teacher(train_image)       # torch.Size([1, 384, 56, 56])
        mean_output = torch.mean(teacher_output, dim=[0, 2, 3])    # torch.Size([384])
        mean_outputs.append(mean_output)
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
    channel_mean = channel_mean[None, :, None, None]

    mean_distances = []
    for train_image, _ in tqdm(train_loader, desc='Computing std of features'):
        if on_gpu:
            train_image = train_image.cuda()
        teacher_output = teacher(train_image)
        distance = (teacher_output - channel_mean) ** 2
        mean_distance = torch.mean(distance, dim=[0, 2, 3])
        mean_distances.append(mean_distance)
    channel_var = torch.mean(torch.stack(mean_distances), dim=0)
    channel_var = channel_var[None, :, None, None]
    channel_std = torch.sqrt(channel_var)

    return channel_mean, channel_std


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='mvtec_ad',
                        choices=['mvtec_ad', 'mvtec_loco'])
    parser.add_argument('-s', '--subdataset', default='bottle',
                        help='One of 15 sub-datasets of Mvtec AD or 5' +
                             'sub-datasets of Mvtec LOCO')
    parser.add_argument('-o', '--output_dir', default='./EfficientAD/output/3')
    parser.add_argument('-m', '--model_size', default='small',
                        choices=['small', 'medium'])
    parser.add_argument('-w', '--weights', default='./EfficientAD/models/teacher_small.pth')
    parser.add_argument('-i', '--imagenet_train_path',
                        default='none',
                        help='Set to "none" to disable ImageNet' +
                             'pretraining penalty. Or see README.md to' +
                             'download ImageNet and set to ImageNet path')
    parser.add_argument('-a', '--mvtec_ad_path',
                        default='./EfficientAD/mvtec_anomaly_detection',
                        help='Downloaded Mvtec AD dataset')
    parser.add_argument('-b', '--mvtec_loco_path',
                        default='./EfficientAD/mvtec_loco_anomaly_detection',
                        help='Downloaded Mvtec LOCO dataset')
    parser.add_argument('-t', '--train_steps', type=int, default=70000)
    parser.add_argument('--cfg', type=str, default='yolov5s_hefei_test.yaml', help='model.yaml')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--yolo_teacher_weights', default='/home/wudidi/code/yolo-distillation-AD/weights/exp13.pt', help='yolo teacher weights')
    return parser.parse_args()

# constants
seed = 42
on_gpu = torch.cuda.is_available()
teacher_out_channels = 64
out_channels = 384
image_size = 256
epochs = 10
h_w_threshold = 32


# data loading
default_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_ae = transforms.RandomChoice([
    transforms.ColorJitter(brightness=0.2),
    transforms.ColorJitter(contrast=0.2),
    transforms.ColorJitter(saturation=0.2)
])

def train_transform(image):
    return default_transform(image), default_transform(transform_ae(image))



def main():
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    config = get_argparse()

    if config.dataset == 'mvtec_ad':
        dataset_path = config.mvtec_ad_path

    pretrain_penalty = True
    if config.imagenet_train_path == 'none':
        pretrain_penalty = False

    # create output dir
    train_output_dir = os.path.join(config.output_dir, 'trainings',
                                    config.dataset, config.subdataset)
    test_output_dir = os.path.join(config.output_dir, 'anomaly_maps',
                                   config.dataset, config.subdataset, 'test')
    if not os.path.exists(train_output_dir):
        os.makedirs(train_output_dir)
        os.makedirs(test_output_dir)

    # load data
    full_train_set = ImageFolderWithoutTarget(
        os.path.join(dataset_path, config.subdataset, 'train'),
        transform=transforms.Lambda(train_transform))
    test_set = ImageFolderWithPath(
        os.path.join(dataset_path, config.subdataset, 'test'))
    if config.dataset == 'mvtec_ad':
        # mvtec dataset paper recommend 10% validation set
        train_size = int(0.9 * len(full_train_set))
        validation_size = len(full_train_set) - train_size
        rng = torch.Generator().manual_seed(seed)
        train_set, validation_set = torch.utils.data.random_split(full_train_set,
                                                           [train_size,
                                                            validation_size],
                                                           rng)

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True,
                              num_workers=4, pin_memory=True)
    train_loader_infinite = InfiniteDataloader(train_loader)
    validation_loader = DataLoader(validation_set, batch_size=1)

    if pretrain_penalty:
        # load pretraining data for penalty
        penalty_transform = transforms.Compose([
            transforms.Resize((2 * image_size, 2 * image_size)),
            transforms.RandomGrayscale(0.3),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
                                                                  0.225])
        ])
        penalty_set = ImageFolderWithoutTarget(config.imagenet_train_path,
                                               transform=penalty_transform)
        penalty_loader = DataLoader(penalty_set, batch_size=1, shuffle=True,
                                    num_workers=4, pin_memory=True)
        penalty_loader_infinite = InfiniteDataloader(penalty_loader)
    else:
        penalty_loader_infinite = itertools.repeat(None)


    if config.model_size == 'small':
        teacher = get_pdn_small(384)
        student = student_model()
    else:
        raise Exception()
    state_dict = torch.load(config.weights, map_location='cpu')
    teacher.load_state_dict(state_dict)
    autoencoder = get_autoencoder(out_channels)

    # teacher frozen
    teacher.eval()
    student.train()
    autoencoder.train()

    if on_gpu:
        teacher.cuda()
        student.cuda()
        autoencoder.cuda()

    teacher_mean, teacher_std = teacher_normalization(teacher, train_loader)

    optimizer = torch.optim.Adam(itertools.chain(student.parameters(),    # 同时更新两个模型参数
                                                 autoencoder.parameters()),
                                 lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(0.95 * config.train_steps), gamma=0.1)
    tqdm_obj = tqdm(range(config.train_steps))


    # create models
    # 添加yolo教师网络
    device = select_device(config.device)    
    # yolo_teacher 加载权重
    w = str(config.yolo_teacher_weights[0] if isinstance(config.yolo_teacher_weights, list) else config.yolo_teacher_weights)
    teacher_yolo = attempt_load(w, map_location=device)
    # 转为eval模式
    teacher_yolo.eval()
    # 获取类别名称
    names = teacher_yolo.module.names if hasattr(teacher_yolo, 'module') else teacher_yolo.names  # get class names
    print("names: ", names)
    # ['traffic-signal-system_good', 'traffic-signal-system_bad', 'traffic-guidance-system_good', 'traffic-guidance-system_bad', 
    #  'restricted-elevated_good', 'restricted-elevated_bad', 
    #  'cabinet_good', 'cabinet_bad', 'backpack-box_good', 'backpack-box_bad', 'off-site', 'Gun-type-Camera', 
    #  'Dome-Camera', 'Flashlight', 'b-Flashlight']
    # 加载数据集    
    source = '/home/wudidi/code/yolo-distillation-AD/datasets/hefei_yolo_format_v2.4/train/images/backpack-box_good*.jpg'
    imgsz=[1280,1280]
    stride = int(teacher_yolo.stride.max())  # model stride
    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    
    val_source = '/home/wudidi/code/yolo-distillation-AD/datasets/hefei_yolo_format_v2.4/val/images/backpack-box_*.jpg'
    val_dataset = LoadImages(val_source, img_size=imgsz, stride=stride)
    for epoch in range(epochs):
        print("================= Start Training Epoch:{} ================== ".format(epoch))
        tqdm_obj = tqdm(range(len(dataset)))
        for iteration, (path, img, im0s, vid_cap) in zip(tqdm_obj, dataset):
            # print(path) # /data/hefei-dataset/hefei_yolo_format_v2.0/train/images/backpack-box_bad_1.jpg            
            _, img_h, img_w = img.shape
            # process image
            img = torch.from_numpy(img).to(device)
            half = False
            img= img.half() if half else  img.float()  # uint8 to fp16/32
            img = img / 255.0  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim
            
            # teacher_yolo inference
            with torch.no_grad():
                pred = teacher_yolo(img, augment=False, visualize=False)
                extract_feature = pred[-1]
                feature = extract_feature[0]
            
            # 找到对应的label文件
            txt_path = str(path).replace('images', 'labels').replace('.jpg', '.txt')
            # 解析标签文件，获得标签信息
            annotations = parse_yolo_annotation(txt_path)
            for annotation in annotations:
                class_id, x, y, width, height = annotation
                
                
                # 针对class_id进行判断，如果id>10, 则跳过
                if int(class_id) != 8:
                    # print("class_id: ", class_id)
                    continue
                
                # 计算边界框的坐标
                x1, y1, x2, y2 = xywh2xyxy(x, y, width, height, img_w, img_h)
                # 过滤掉一些不符合要求的边界框
                if is_ignore(x1, x2, y1, y2, threshold=h_w_threshold):
                    continue
                
                # 获取crop后的feature和image
                cropped_feature, cropped_img = get_crop(x1, y1, x2, y2, feature, img)
                # print("cropped_feature shape: ", cropped_feature.shape)  # torch.Size([1, 384, 56, 56])
                # 对裁剪的feature 进行正则化
                teacher_output_st = (cropped_feature - teacher_mean[:,:teacher_out_channels,:,:]) / teacher_std[:,:teacher_out_channels,:,:]
                
                
                # 开始训练学生模型
                student_output_st = student(cropped_img)[:, :teacher_out_channels]  #  torch.Size([1, 768, 56, 56])
                # print("student_output_st shape: ", student_output_st.shape)
                distance_st = (teacher_output_st - student_output_st) ** 2
                d_hard = torch.quantile(distance_st, q=0.999)
                loss_hard = torch.mean(distance_st[distance_st >= d_hard])
                loss_total = loss_hard
                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()
                scheduler.step()
                
                if iteration % 10 == 0:
                    tqdm_obj.set_description(
                        "Current loss: {:.4f}  ".format(loss_total.item()))
                
                if iteration % 1000 == 0:
                    torch.save(teacher_yolo, os.path.join(train_output_dir,
                                                    'teacher_tmp.pth'))
                    torch.save(student, os.path.join(train_output_dir,
                                                    'student_tmp.pth'))
                    
                if iteration % 5000 == 0 and iteration > 0:
                    teacher_yolo.eval()
                    student.eval()
                    
                    qq_st_start, qq_st_end = my_map_normalization(
                        validation_loader=val_dataset, teacher=teacher_yolo, student=student,
                        teacher_mean=teacher_mean, teacher_std=teacher_std,
                        desc='Intermediate map normalization', device=device)
                    
                    auc = my_test(
                        test_set=val_dataset, teacher=teacher_yolo, student=student,
                        teacher_mean=teacher_mean, teacher_std=teacher_std, q_st_start=qq_st_start,
                        q_st_end=qq_st_end,
                        test_output_dir=None, desc='Intermediate inference',device=device)
                    print('Intermediate image auc: {:.4f}'.format(auc))
                    
                    student.train()
        
        # 一轮训练结束，进行一次验证        
        if True:
            teacher_yolo.eval()
            student.eval()
            
            qq_st_start, qq_st_end = my_map_normalization(
                validation_loader=val_dataset, teacher=teacher_yolo, student=student,
                teacher_mean=teacher_mean, teacher_std=teacher_std,
                desc='Intermediate map normalization', device=device)
            
            auc = my_test(
                test_set=val_dataset, teacher=teacher_yolo, student=student,
                teacher_mean=teacher_mean, teacher_std=teacher_std, q_st_start=qq_st_start,
                q_st_end=qq_st_end,
                test_output_dir=None, desc='Intermediate inference',device=device)
            print('Intermediate image auc: {:.4f}'.format(auc))
            torch.save(teacher_yolo, os.path.join(train_output_dir,
                                                    f'teacher_{epoch}.pth'))
            torch.save(student, os.path.join(train_output_dir,
                                                    f'student_{epoch}.pth'))
            student.train()
                    
    torch.save(teacher, os.path.join(train_output_dir, 'teacher_final.pth'))
    torch.save(student, os.path.join(train_output_dir, 'student_final.pth'))
    torch.save(autoencoder, os.path.join(train_output_dir,
                                         'autoencoder_final.pth'))


if __name__ == '__main__':
    main()
