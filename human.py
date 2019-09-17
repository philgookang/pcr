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

if __name__ == "__main__":

    # load test & train dataset
    # test_dataset = load_dataset(dataset_file["test"])
    # test_dataset = load_dataset("cnn_rnn_test_dataset_flickr8k.pkl")
    test_dataset = load_dataset("cnn_rnn_test_dataset.pkl")

    # result holder
    result_holder = []

    # skip count
    skip_count = 0

    # create image yes or no
    should_create_image = True

    # the loading bar
    for filename in tqdm(test_dataset):

        # list of referece sentence
        test_list = test_dataset[filename]

        ref1 = test_list[0]
        ref2 = test_list[1]
        ref3 = test_list[2]
        ref4 = test_list[3]
        ref5 = test_list[4]

        # add list
        result_holder.append({
            "reference"  : [ref2, ref3, ref4, ref5],
            "hypothesis" : ref1
        })

    # test human value
    run_score(result_holder)
