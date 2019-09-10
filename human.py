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
    test_dataset = load_dataset(dataset_file["test"])

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

        # add list
        result_holder.append({
            "reference" : test_list[1:],
            "hypothesis" : test_list[0]
        })

    # test human value
    run_score(result_holder)
