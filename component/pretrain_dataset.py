import torch
import torch.utils.data as data
import os
from PIL import Image

from config import *
from helper import *

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings

class PretrainDataset(data.Dataset):

    def __init__(self, **kwargs):
        self.dataset = kwargs["dataset"]
        self.transform = kwargs["transform"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(self.dataset["corpus"])
            convert = self.label_encoder.transform(self.dataset["corpus"])
            convert = convert.reshape(len(convert), 1)

            self.one_hot_encoder = OneHotEncoder(sparse = False)
            self.one_hot_encoder.fit(convert)


    def __len__(self):
        return len(self.dataset["data"])


    def __getitem__(self, index):

        # get data item from dataset
        item = self.dataset["data"][index]

        # get image
        image = self.load_image(coco_pretrain_image_path + item["filename"])

        # get one hot encoding of word
        label = self.label_encoder.transform([item["word"]])
        label = self.one_hot_encoder.transform([label])
        label = torch.from_numpy(label[0])

        return image, label


    def load_image(self, filename):
        image = Image.open(filename).convert("RGB")
        image = self.transform(image)
        return image
