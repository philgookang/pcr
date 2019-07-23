import pickle
import torch

from component import *
from helper import *
from model import *
from PIL import Image

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings

import torch.nn as nn
from torchvision import transforms

class Pcr():

    def __init__(self, **kwargs):
        self.save_object_path = kwargs["model_path"]
        self.noun_model = None
        self.rnn_model = None

        self.train_dataset = load_dataset(dataset_file["train"])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(self.train_dataset["corpus"])

    def load_cnn_model(self, pos, embed_size, device):
        features = torch.load("result/model/" + model_file[pos]["pretrain"])
        cnn_model = Encoder(embed_size = embed_size)
        cnn_model = nn.DataParallel(cnn_model)
        cnn_model.to(device, non_blocking=True)
        cnn_model.load_state_dict(features)
        return cnn_model

    def load_rnn_model(self, embed_size, hidden_size, output_size, number_of_layers):
        features = torch.load("result/model/" + model_file["decoder"]["train"])
        decoder_model = Decoder( embed_size, hidden_size, output_size, number_of_layers)
        decoder_model = nn.DataParallel(decoder_model, device_ids = [0])
        decoder_model.load_state_dict(features)
        return decoder_model

    def load_image_file(self, filename_with_full_path, transform = None):
        image  = Image.open(filename_with_full_path)
        image = image.resize([224, 224], Image.LANCZOS)
        if transform is not None:
            image = transform(image).unsqueeze(0)
        return image


    def testcase(self, image_file_full_path):

        # ############################################################
        # VARIABLES & LOADING
        # ############################################################

        # GPU or CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # output size of CNN and input size of RNN
        embed_size = 256

        # hidden size
        hidden_size = 512

        # number of layers
        number_of_layers = 1

        # image preprocessing
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        # load cnn model
        if self.noun_model == None:
            self.noun_model      = self.load_cnn_model("noun", embed_size, device)
            self.verb_model      = self.load_cnn_model("verb", embed_size, device)
            self.adjective_model = self.load_cnn_model("adjective", embed_size, device)
            self.conjunction_model = self.load_cnn_model("conjunction", embed_size, device)
            self.preposition_model = self.load_cnn_model("preposition", embed_size, device)

        # load rnn model
        if self.rnn_model == None:
            self.rnn_model = self.load_rnn_model((embed_size * 5), hidden_size, len(train_dataset["corpus"]), number_of_layers)

        # load image
        image = self.load_image_file(image_file_full_path, transform)


        # ############################################################
        # MODEL PROCESSING
        # ############################################################

        # send all objects to GPU
        self.noun_model = self.noun_model.to(device, non_blocking = True)
        self.verb_model = self.verb_model.to(device, non_blocking = True)
        self.adjective_model = self.adjective_model.to(device, non_blocking = True)
        self.conjunction_model = self.conjunction_model.to(device, non_blocking = True)
        self.preposition_model = self.preposition_model.to(device, non_blocking = True)
        self.rnn_model = self.rnn_model.to(device, non_blocking = True)
        self.rnn_model.module = self.rnn_model.module.to(device, non_blocking = True)
        image = image.to(device, non_blocking = True)

        # change model mode to eval
        self.noun_model = self.noun_model.eval()
        self.verb_model = self.verb_model.eval()
        self.adjective_model = self.adjective_model.eval()
        self.conjunction_model = self.conjunction_model.eval()
        self.preposition_model = self.preposition_model.eval()

        # detect
        # encoder
        noun_features = self.noun_model(image)
        verb_features = self.verb_model(image)
        adjective_features = self.adjective_model(image)
        conjunction_features = self.conjunction_model(image)
        preposition_features = self.preposition_model(image)
        features = combine_vertically(noun_features, verb_features, adjective_features, conjunction_features, preposition_features)
        features = features.to(device, non_blocking = True)

        # decoder
        sampled_ids = self.rnn_model.module.sample(features)
        sampled_ids = sampled_ids[0].cpu().numpy()

        # # reverse label encoding
        sampled_caption = []
        for word_id in sampled_ids:
            inverted_label = self.label_encoder.inverse_transform([word_id])
            word = inverted_label[0]
            sampled_caption.append(word)
            if word == '<end>':
                break

        return sampled_caption[1:-1]