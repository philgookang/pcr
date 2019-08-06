from component import *
from config import *
from model import *
from helper import *

import os
import pickle
import torch
import numpy as np
import torch.nn as nn
import visdom
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence
from shutil import copyfile
from tqdm import tqdm

from PIL import Image

def heatmap_cnn_image(filename, file_name_to_export, device, pos, target_class, target_layer):

    # load image
    image = Image.open(filename).convert('RGB')

    features = torch.load(model_save_path + model_file[pos]["train"])
    cnn_model = Encoder(embed_size = 256)
    cnn_model = nn.DataParallel(cnn_model)
    cnn_model.to(device, non_blocking=True)
    cnn_model.load_state_dict(features)
    cnn_model.eval() # change model mode to eval

    cnn_model = cnn_model.module
    cnn_model.update_layer(len(dataset["corpus"]))
    cnn_model.eval()

    # Grad cam
    grad_cam = GradCam(cnn_model, target_layer = target_layer)

    # Prepare image
    prep_img = grad_cam.preprocess_image(image)

    # Generate cam mask
    cam = grad_cam.generate_cam(prep_img, target_class)

    # save result
    grad_cam.save_class_activation_images(image, cam, file_name_to_export)

if __name__ == "__main__":

    pos = "noun"
    dataset = load_dataset(dataset_file["pretrain"])
    dataset = dataset[pos]

    trans =  transforms.Compose([transforms.RandomCrop(image_crop_size),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    pretrain_dataset = PretrainDataset(dataset = dataset, transform = trans)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target_layer = 4

    for i in tqdm(range(len(pretrain_dataset))):
        item = pretrain_dataset.get(i)

        label = pretrain_dataset.convert_word(item["word"])

        heatmap_org_image_result_path = os.path.join(heatmap_cnn_path, pos, item["word"] + "_" + item["filename"].replace(".", "_o.") )
        heatmap_result_path = os.path.join(heatmap_cnn_path, pos, item["word"] + "_" + item["filename"].replace(".", "_" + item["word"] + ".") )

        copyfile(coco_pretrain_image_path_original + item["filename"], heatmap_org_image_result_path)
        heatmap_cnn_image(coco_pretrain_image_path + item["filename"], heatmap_result_path, device, pos, label, target_layer)
