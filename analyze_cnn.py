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
import nltk

from PIL import Image

import sys

def load_ml(ds, pos, type, trans, device):

    pretrain_dataset = PretrainDataset(dataset = ds, transform = trans, img_path = IMG_PATH)

    features = torch.load(RESULT_MODEL_PATH + model_file[pos][type])
    cnn_model = Encoder(embed_size = len(ds["corpus"]))
    cnn_model = nn.DataParallel(cnn_model) # , device_ids = [gpu_num]
    cnn_model.to(device, non_blocking=True)
    cnn_model.load_state_dict(features)
    cnn_model.eval() # change model mode to eval

    return pretrain_dataset, cnn_model

def load_image_file(filename, trans, device):
    image  = Image.open(IMG_PATH + filename)
    image = image.resize([224, 224], Image.LANCZOS)
    image = trans(image).unsqueeze(0)
    image = image.to(device, non_blocking = True)
    return image

def convert_to_pos(reference_list):
    lst = []
    for sentence in reference_list:
        lst.extend(sentence)
        # tokens_pos = nltk.pos_tag(sentence)
    return lst


pos = 'verb' # <- change this!
type = "pretrain"

device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trans   =  transforms.Compose([transforms.RandomCrop(image_crop_size),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

pretrainset = load_dataset(dataset_file["pretrain"])
testset = load_dataset(dataset_file["test"])

loader, model = load_ml(pretrainset[pos], pos, type, trans, device)


cnt = 0
total_cnt = 0
skip_cnt = 0

for filename in tqdm(testset):

    image = load_image_file(filename, trans, device)

    features = None
    try:
        features = model(image)
    except:
        continue

    word = loader.decode_word(features)

    postag = nltk.pos_tag([word])[0]
    if postag[1] not in nltk_to_pos[pos]:
        skip_cnt += 1
        continue

    word_lst = convert_to_pos(testset[filename])

    if  word in word_lst:
        cnt += 1

    total_cnt += 1

print("cnt", cnt)
print("total_cnt", total_cnt)
print("skip_cnt", skip_cnt)

# 2577 / 4677 - verb 55.09
# 4965 / 4987 - adjective 99.55
# 4209 /  4987 - conjunction 84.39
# 4224 / 4986 - preposition 84.71
