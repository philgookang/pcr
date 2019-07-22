import torch
import torch.utils.data as data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings

from config import *
from helper import *

class TrainDataset(data.Dataset):

    def __init__(self, **kwargs):
        self.dataset = kwargs["dataset"]
        self.transform = kwargs["transform"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(self.dataset["corpus"])


    def __len__(self):
        return len(self.dataset["data"])


    def __getitem__(self, index):

        # get data item from dataset
        item = self.dataset["data"][index]

        # get image
        image = load_image(coco_image_path + item["filename"])

        # get one hot encoding of word
        label = get_label(item["caption"])

        return image, label


    def convert_word(self, word):
        label = self.label_encoder.transform([word])
        return label[0]


    def get_label(self, item):

        caption = [ ]
        caption.append(self.convert_word("<start>"))
        caption.extend([self.convert_word(word) for word in item["caption"]])
        caption.append(self.convert_word("<end>"))
        target = torch.Tensor(caption)

        return target
