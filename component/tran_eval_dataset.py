import torch
import torch.utils.data as data
from PIL import Image

from sklearn.preprocessing import LabelEncoder
import warnings

class TrainEvalDataset(data.Dataset):

    def __init__(self, **kwargs):
        self.dataset = kwargs["dataset"]
        self.transform = kwargs["transform"]
        self.image_path = kwargs["image_path"]

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
        image = self.load_image(self.image_path + item["filename"])

        # get one hot encoding of word
        label = self.get_label(item["caption"])

        return image, label


    def convert_word(self, word):
        label = self.label_encoder.transform([word])
        return label[0]


    def get_label(self, word_list):

        caption = [ ]
        caption.append(self.convert_word("<start>"))
        for word in word_list:
            caption.append(self.convert_word(word))
        caption.append(self.convert_word("<end>"))
        target = torch.Tensor(caption)

        return target

    def load_image(self, filename):
        image = Image.open(filename).convert("RGB")
        image = self.transform(image)
        return image
