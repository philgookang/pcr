import pickle
import torch
import os

from config import *
from component import *
from helper import *
from model import *
from PIL import Image

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings

import torch.nn as nn
from torchvision import transforms

from score import run_score
from tqdm import tqdm

from resize_image import resize_images
from test import Pcr

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--targets', type=str, default=RESULT_ROOT + '/live/targets/' , help='')
    parser.add_argument('--resize', type=str, default=RESULT_ROOT + '/live/resize/' , help='')
    parser.add_argument('--results', type=str, default=RESULT_ROOT + '/live/results/' , help='')
    args = parser.parse_args()

    print("resize test images")
    resize_images(args.targets, args.resize, [256, 256])

    print("load corpus dataset")
    train_dataset = load_dataset(dataset_file["train"])

    # create PCR
    # skip count
    print("loading PCR model")
    pcr = Pcr(train_dataset)
    skip_count = 0
    process_count = 0

    # the loading bar
    # create hypothesis
    print("begin caption generation")
    for filename in tqdm(os.listdir(args.resize)):
        try:
            hypothesis = pcr.testcase(os.path.join(args.resize + filename))
            create_image_caption(os.path.join(args.targets, filename), os.path.join(args.results,  filename), [' '.join(hypothesis)])
            process_count += 1
        except:
            skip_count += 1

    print("end caption generation")
    print("process_count:", process_count)
    print("skip_count:", skip_count)
