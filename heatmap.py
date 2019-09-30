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

import sys

def heatmap_cnn_image(cnn_model, filename, file_name_to_export, device, pos, target_class, target_layer, dataset):

    # load image
    image = Image.open(filename).convert('RGB')

    # Grad cam
    grad_cam = GradCam(cnn_model, target_layer = target_layer)

    # Prepare image
    prep_img = grad_cam.preprocess_image(image)

    # Generate cam mask
    cam = grad_cam.generate_cam(prep_img, target_class)

    # save result
    grad_cam.save_class_activation_images(image, cam, file_name_to_export)

def load_image_file(filename_with_full_path, trans, device):
    image  = Image.open(filename_with_full_path)
    image = image.resize([224, 224], Image.LANCZOS)
    image = trans(image).unsqueeze(0)
    image = image.to(device, non_blocking = True)
    return image

if __name__ == "__main__":

    if len(sys.argv) >= 2:

        # get filename
        type = sys.argv[1]

        print("type", type)

        test_dataset = load_dataset(dataset_file["test"])
        dataset = load_dataset(dataset_file["pretrain"])

        trans =  transforms.Compose([transforms.RandomCrop(image_crop_size),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        target_layer = 4

        pos = "verb"

        pretrain_dataset = PretrainDataset(dataset = dataset[pos], transform = trans, img_path = IMG_PATH)

        features = torch.load(RESULT_MODEL_PATH + model_file[pos]["train"])
        cnn_model = Encoder(embed_size = 1024)
        cnn_model = nn.DataParallel(cnn_model) # , device_ids = [gpu_num]
        cnn_model.to(device, non_blocking=True)
        cnn_model.load_state_dict(features)
        cnn_model.eval() # change model mode to eval

        cnn_model = cnn_model.module
        cnn_model.update_layer(len(dataset[pos]["corpus"]))
        cnn_model.eval()
        cnn_model.to(device, non_blocking=True)

        i = 0
        for filename in tqdm(test_dataset):

            image = load_image_file(IMG_PATH + filename, trans, device)

            features = None
            try:
                features = cnn_model(image)
            except RuntimeError:
                continue

            word = pretrain_dataset.decode_word(features)
            label = pretrain_dataset.convert_word(word)

            # create file name
            file_split = filename.split('.')
            newfilename = "{0}_{1}.{2}".format(i, word, file_split[1])

            # get image
            heatmap_org_image_result_path = os.path.join(HEATMAP_CNN, pos, newfilename.replace(".", "_o.") )
            heatmap_result_path = os.path.join(HEATMAP_CNN, pos, newfilename )

            copyfile(IMG_PATH + filename, heatmap_org_image_result_path)
            heatmap_cnn_image(cnn_model, IMG_PATH + filename, heatmap_result_path, device, pos, label, target_layer, dataset[pos])


            i += 1




#
